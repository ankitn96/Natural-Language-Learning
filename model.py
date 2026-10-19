"""Model definition for the CITS4012 PIQA GoalConditionedAttentionBiLSTM.

This module recreates the architecture used in the training notebook so a saved
state_dict can be loaded without rerunning training. Keep this file in sync with
the model definition used to produce the checkpoint.
"""

import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_softmax(scores, mask, dim=-1):
    """Apply softmax over valid positions only; padded positions get zero weight."""
    mask = mask.bool()
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-9)


class SharedBiLSTMEncoder(nn.Module):
    """Encode token sequences with a shared embedding and bidirectional LSTM."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_size=64,
                 pad_id=0, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size,
            batch_first=True, bidirectional=True
        )
        self.output_dim = hidden_size * 2

    def forward(self, ids, mask):
        lengths = mask.long().sum(dim=1).clamp_min(1).cpu()
        embedded = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        hidden, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=ids.size(1)
        )
        return self.dropout(hidden)


class GoalConditionedAttentionBiLSTM(nn.Module):
    """Shared BiLSTM with learned pooling attention and goal-to-solution cross-attention."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_size=64,
                 pad_id=0, dropout=0.2,
                 use_self_attention=True, use_cross_attention=True):
        super().__init__()
        self.encoder = SharedBiLSTMEncoder(
            vocab_size, embedding_dim, hidden_size, pad_id, dropout
        )
        d = self.encoder.output_dim
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.goal_pool_score = nn.Linear(d, 1)
        self.solution_pool_score = nn.Linear(d, 1)
        self.query_projection = nn.Linear(d, d, bias=False)
        self.key_projection = nn.Linear(d, d, bias=False)
        self.value_projection = nn.Linear(d, d, bias=False)
        self.goal_fusion = nn.Linear(2 * d, d)
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )
        self.scale = math.sqrt(d)

    def pool(self, hidden, mask, scorer):
        if self.use_self_attention:
            weights = masked_softmax(
                scorer(hidden).squeeze(-1), mask, dim=1
            )
        else:
            weights = (
                mask.to(hidden.dtype)
                / mask.to(hidden.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
            )
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    def score_one(self, goal_ids, goal_mask, sol_ids, sol_mask,
                  need_attention=False):
        goal_hidden = self.encoder(goal_ids, goal_mask)
        sol_hidden = self.encoder(sol_ids, sol_mask)

        if self.use_cross_attention:
            q = self.query_projection(goal_hidden)
            k = self.key_projection(sol_hidden)
            v = self.value_projection(sol_hidden)
            cross_scores = torch.matmul(q, k.transpose(1, 2)) / self.scale
            cross_weights = masked_softmax(
                cross_scores, sol_mask.unsqueeze(1), dim=-1
            )
            cross_context = torch.matmul(cross_weights, v)
            goal_enriched = torch.tanh(
                self.goal_fusion(
                    torch.cat([goal_hidden, cross_context], dim=-1)
                )
            )
        else:
            cross_weights = goal_hidden.new_zeros(
                goal_hidden.size(0), goal_hidden.size(1), sol_hidden.size(1)
            )
            goal_enriched = goal_hidden

        goal_vec, goal_weights = self.pool(
            goal_enriched, goal_mask, self.goal_pool_score
        )
        sol_vec, sol_weights = self.pool(
            sol_hidden, sol_mask, self.solution_pool_score
        )
        features = torch.cat(
            [goal_vec, sol_vec, torch.abs(goal_vec - sol_vec), goal_vec * sol_vec],
            dim=-1
        )
        score = self.scorer(self.dropout(features)).squeeze(-1)

        if need_attention:
            return score, {
                "goal_self": goal_weights,
                "solution_self": sol_weights,
                "cross": cross_weights
            }
        return score

    def forward(self, batch, return_attention=False):
        if return_attention:
            score1, att1 = self.score_one(
                batch["goal_ids"], batch["goal_mask"],
                batch["sol1_ids"], batch["sol1_mask"], True
            )
            score2, att2 = self.score_one(
                batch["goal_ids"], batch["goal_mask"],
                batch["sol2_ids"], batch["sol2_mask"], True
            )
            return (
                torch.stack([score1, score2], dim=1),
                {"candidate1": att1, "candidate2": att2}
            )

        score1 = self.score_one(
            batch["goal_ids"], batch["goal_mask"],
            batch["sol1_ids"], batch["sol1_mask"]
        )
        score2 = self.score_one(
            batch["goal_ids"], batch["goal_mask"],
            batch["sol2_ids"], batch["sol2_mask"]
        )
        return torch.stack([score1, score2], dim=1)

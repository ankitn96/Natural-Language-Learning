# PIQA Physical Commonsense Reasoning with Deep Learning

**A comparative study of recurrent neural networks, attention mechanisms, Transformer models, and BERT for physical commonsense reasoning.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch)
![NLP](https://img.shields.io/badge/Domain-NLP-purple)
![Status](https://img.shields.io/badge/Status-Completed%20Experiments%20%2F%20Update%20as%20Needed-informational)

## Project Overview

How can a machine determine which of two possible solutions is more sensible in the physical world?

This project explores **physical commonsense reasoning using Natural Language Processing (NLP)** through the PIQA (Physical Interaction: Question Answering) benchmark. Given a goal and two candidate solutions, a model must identify the solution that is most physically plausible.

Unlike tasks that can be solved through surface-level text matching alone, physical commonsense reasoning requires a model to learn relationships between actions, objects, and their intended outcomes.

We developed and evaluated multiple neural-network architectures, including models trained from scratch and a fine-tuned BERT baseline. The experiments investigate whether recurrent sequence modelling and attention mechanisms improve the selection of physically plausible solutions.

The project combines model development, hyperparameter tuning, controlled ablation studies, quantitative evaluation, and prediction visualisation to better understand the strengths and limitations of different approaches.

## Project Objectives

- Develop a custom neural-network model for physical commonsense reasoning.
- Establish baseline performance using BiLSTM, Transformer, and fine-tuned BERT architectures.
- Investigate self-attention and goal-conditioned cross-attention for modelling relationships between a goal and its candidate solutions.
- Conduct controlled ablation experiments to evaluate the contribution of individual attention mechanisms.
- Tune training hyperparameters using validation data and early stopping.
- Compare model performance using accuracy, macro-precision, macro-recall, macro-F1, and test loss.
- Save trained model artefacts to support reproducible inference without retraining.

## Model Architectures

The project evaluates complementary approaches to understanding the relationship between a question's goal and its possible solutions.

### 1. BiLSTM Baseline

A bidirectional Long Short-Term Memory network encodes the goal and candidate solutions using a shared encoder. Masked mean pooling produces fixed-length representations, which are combined using interaction features before a shared scoring layer ranks the candidates.

**Purpose:** Establish a recurrent neural-network baseline trained from scratch.

### 2. Transformer Baseline

A Transformer encoder with trainable token and positional embeddings processes concatenated goal–candidate sequences.

**Purpose:** Evaluate attention-based sequence modelling without relying on pretrained language-model weights.

### 3. Main Model — Attention-Based BiLSTM

The main scratch model combines a shared BiLSTM encoder with two attention mechanisms:

- **Self-attention pooling:** Learns which token representations are more useful when summarising the goal and candidate solutions.
- **Goal-conditioned cross-attention:** Allows representations of the goal to attend to the tokens in a candidate solution, modelling relationships between the intended outcome and the proposed action.

The resulting representations are combined using interaction features and passed to a shared scoring network. The candidate with the higher score is selected.

**Purpose:** Investigate whether explicitly modelling goal–solution relationships improves physical commonsense reasoning.

### 4. Attention Ablation Studies

Ablation experiments remove one attention mechanism at a time while retaining the rest of the main architecture.

| Experiment | Self-attention pooling | Cross-attention |
|---|---|---|
| Full attention model | Enabled | Enabled |
| No self-attention | Disabled | Enabled |
| No cross-attention | Enabled | Disabled |

These comparisons help isolate the contribution of each component rather than attributing any performance difference to the overall architecture alone.

### 5. Fine-Tuned BERT Baseline

A pretrained BERT sequence-classification model is fine-tuned on the PIQA training data using goal–candidate text pairs.

**Purpose:** Provide a pretrained-language-model reference point against which the custom, from-scratch models can be evaluated.

> **Experimental distinction:** The custom BiLSTM and Transformer architectures are trained from scratch. BERT uses pretrained parameters and is fine-tuned for the task.

## Experimental Methodology

### Data preparation

The project uses the PIQA dataset, containing goals, alternative solutions, and answer labels. The data is prepared for the appropriate input format for each architecture.

The scratch models use a vocabulary derived from the training partition, with padding and unknown-token handling. BERT uses its corresponding pretrained tokenizer.

### Training and model selection

- AdamW optimiser for the scratch models.
- Cross-entropy loss for the two-candidate classification task.
- Gradient clipping to stabilise optimisation.
- Early stopping based on validation performance.
- Hyperparameter search over selected learning rates and patience values for the main scratch model.
- Validation-based checkpoint selection, with validation loss used to break accuracy ties in the grid-search implementation.

The held-out test set is intended for final evaluation rather than hyperparameter selection.

### Evaluation metrics

| Metric | What it measures |
|---|---|
| Accuracy | Proportion of candidate choices predicted correctly |
| Macro precision | Average precision across the two answer classes |
| Macro recall | Average recall across the two answer classes |
| Macro-F1 | Harmonic mean of macro precision and macro recall |
| Weighted-F1 | F1 averaged according to class frequency |
| Test loss | Cross-entropy loss on the held-out test examples |

The analysis also includes performance-comparison plots, confusion matrices, training-history plots, and attention visualisations where supported by the model implementation.

## Results

*Populate this section with the final results produced by your notebook. Do not report validation metrics as test metrics.*

### Model comparison

| Model | Test accuracy | Macro-F1 | Test loss |
|---|---:|---:|---:|
| Main attention BiLSTM | 0.5838 | 0.5838 | 0.7714 |
| BiLSTM baseline | 0.5658 | 0.5656 | 0.7890 |
| Transformer baseline | 0.5392 | 0.5392 | 0.6926 |
| BiLSTM without self-attention | 0.5811 | 0.5810 | 0.7991 |
| BiLSTM without cross-attention | 0.5702 | 0.5702 | 0.7014 |
| Fine-tuned BERT baseline | 0.6480 | 0.6480 | 0.9502 |


### Key findings

Add a concise interpretation of your actual results here:

- **Best-performing model:** [Insert model name and test metrics.]
- **Attention ablation findings:** [Describe how removing self-attention or cross-attention affected validation and test performance.]
- **BERT comparison:** [Explain how the fine-tuned pretrained baseline compared with the custom models.]
- **Limitations:** [Discuss generalisation, dataset size, training resources, or observed failure cases.]

Performance differences should be interpreted in the context of the experimental setup. A higher score in one run does not, by itself, establish that an architectural component caused the improvement.

## Example Prediction

The model accepts a goal and two candidate solutions and returns its predicted choice.

**Illustrative input**

- **Goal:** Keep your hands warm in cold weather.
- **Solution 1:** Wear insulated gloves.
- **Solution 2:** Dip your hands in ice water.

**Expected sensible choice:** Solution 1.

*This is an illustrative example, not a recorded prediction from the trained model. Add the actual output from your inference function when available.*

## Repository Structure

```text
cits4012-piqa-model/
├── README.md
├── model.py
├── main_attention_best.pt
├── main_attention_config.json
├── scratch_vocab.json
├── inference.ipynb
└── requirements.txt
```

The exact repository structure may differ depending on which notebooks, scripts and model artefacts are published.

- `model.py` — custom main-model architecture.
- `main_attention_best.pt` — trained main-model weights and checkpoint metadata.
- `main_attention_config.json` — architecture configuration for reconstruction.
- `scratch_vocab.json` — vocabulary used by the scratch model.
- `inference.ipynb` — example notebook for loading the saved model and making predictions.
- `requirements.txt` — Python dependencies and compatible versions.

Large datasets and unnecessary intermediate training files should generally be kept out of the public repository. The pretrained BERT checkpoint can be stored separately if needed.

## Running Inference

The trained main model can be loaded without repeating the training or grid-search process, provided the checkpoint, architecture definition, configuration and vocabulary are available.

1. Clone or download this repository.
2. Install the dependencies listed in `requirements.txt`.
3. Load `model.py` and reconstruct the main model using `main_attention_config.json`.
4. Restore the saved weights from `main_attention_best.pt`.
5. Load `scratch_vocab.json` and apply the same preprocessing used during training.
6. Pass a goal and two candidate solutions to the prediction function.

See [`inference.ipynb`](inference.ipynb) for the implementation.

**Reproducibility note:** The vocabulary and preprocessing logic must match the original training pipeline. Saved weights alone are not sufficient to reproduce predictions.

## Technologies Used

- **Python** — implementation and experimentation
- **PyTorch** — neural-network development and training
- **Natural Language Processing** — tokenisation and text representation
- **BiLSTM networks** — bidirectional sequence encoding
- **Transformer encoders** — attention-based sequence modelling
- **Self-attention and cross-attention** — token weighting and goal–solution interaction
- **BERT / Hugging Face Transformers** — pretrained baseline and fine-tuning
- **Scikit-learn** — evaluation metrics and confusion matrices
- **Pandas, NumPy and Matplotlib** — results analysis and visualisation
- **Google Colab / local Python environment** — experimentation and inference

## Team and Contributors

This project was developed collaboratively as part of the CITS4012 Natural Language Processing coursework.

*Contributor details will be added here.*

| Contributor | Role and contributions | Profile |
|---|---|---|
| [Your name] | [Model development, experimentation, evaluation, or other contributions] | [GitHub / LinkedIn] |
| [Co-contributor 2] | [Contribution summary] | [GitHub / LinkedIn] |
| [Co-contributor 3] | [Contribution summary] | [GitHub / LinkedIn] |
| [Co-contributor 4] | [Contribution summary] | [GitHub / LinkedIn] |

Update the table to reflect the actual team and each member's contributions.

## Future Improvements

Potential directions for further work include:

- Expanding hyperparameter tuning and evaluating robustness across multiple random seeds.
- Investigating alternative pooling mechanisms and richer goal–solution interaction layers.
- Analysing incorrect predictions to identify common physical-reasoning failure modes.
- Comparing additional pretrained language models under a consistent evaluation protocol.
- Improving inference efficiency and packaging the trained model as a simple interactive application.
- Evaluating whether attention visualisations provide useful explanations of the model's decisions.

## Acknowledgements

This work was completed as part of the CITS4012 Natural Language Processing unit at the University of Western Australia. The project uses the PIQA benchmark for physical commonsense reasoning.

Please add any required dataset, model, course, and third-party software citations here, in accordance with the assignment requirements.

## License

[Choose and specify a licence before publishing the repository for reuse. Confirm that the dataset and any included model artefacts can be redistributed under the relevant terms.]

---

**Project focus:** Building, comparing and interpreting neural models for physical commonsense reasoning, with particular emphasis on attention mechanisms, controlled ablation experiments, and reproducible model inference.


# Semi-Supervised Temporal Modeling with First-Principles Intervals and a Mixture of Kumaraswamy Distributions

This repository provides a public implementation of the method described in our paper on semi-supervised temporal modeling for sparse-sampling industrial processes. The method integrates first-principles interval construction with a neural network that outputs a mixture of truncated Kumaraswamy distributions.

---

## Overview

In many industrial processes, labels are sampled much more sparsely than process variables. To address this issue, this repository implements a semi-supervised temporal modeling framework with the following key ideas:

- **First-principles models** are used to construct **label intervals** rather than point predictions.
- A data-driven temporal model is trained to learn the **label location within these intervals**.
- The predictive distribution is represented as a **mixture of truncated Kumaraswamy distributions**.
- Both **labeled** and **unlabeled** samples are used during training.

For demonstration purposes, this repository also includes a synthetic dataset generator. In the toy example, the label intervals are not constructed by directly perturbing the labels. Instead, an interval is first assigned to one variable in the label-generation formula, and the corresponding label interval is then obtained by propagating this variable interval through the same formula. Two kinds of intervals are generated in this way:

- **WFPI**: a wider interval corresponding to a weaker first-principles constraint
- **SFPI**: a narrower interval corresponding to a stronger first-principles constraint

The repository currently contains five main Python files:

- `Method.py` – the semi-supervised neural model and training procedure  
- `IndustrialCase_WFPI.py` – weak first-principles interval construction for industrial case in paper
- `IndustrialCase_SFPI.py` – strong first-principles interval construction for industrial case in paper
- `ToyDataSet.py` – synthetic dataset generation and toy interval construction  
- `Run.py` – a runnable demo script for training, evaluation, and visualization

---

## Method.py Interface

`Method.py` provides the main training and prediction interface of the proposed semi-supervised model. It contains two key components:

- `MethodConfig` – a configuration class for specifying the model architecture and training hyperparameters
- `fit_predict` – the main function for model training and test-set prediction

### 1. `MethodConfig`

`MethodConfig` is used to define the architectural settings and training hyperparameters of the proposed method. It allows users to adjust the main options of the semi-supervised model without modifying the model source code directly.

Typical configurable items include:

- number of mixture components
- hidden dimension and number of LSTM layers for the **weight branch**
- hidden dimension and number of LSTM layers for the **component branches**
- convolution settings, such as kernel size, stride, and padding
- labeled and unlabeled batch sizes
- number of training epochs
- learning rate and weight decay
- semi-supervised loss weight
- prediction interval significance level
- device setting (`cpu` or `cuda`)

A typical example is:

```python
from Method import MethodConfig

cfg = MethodConfig(
    n_components=2,
    weight_lstm_hidden_dim=32,
    weight_lstm_layers=1,
    component_lstm_hidden_dim=32,
    component_lstm_layers=1,
    conv_kernel_size=(3, 1),
    conv_stride=(1, 1),
    conv_padding=(1, 0),
    labeled_batch_size=32,
    unlabeled_batch_size=64,
    lambda_u=0.3,
    q_s=0.9,
    alpha=0.1,
    epochs=120,
    learning_rate=1e-3,
    weight_decay=1e-5,
    device="cpu",
    verbose=True,
)

---

## Repository Structure

```text
.
├── Method.py      # Semi-supervised temporal model
├── IndustrialCase_WFPI.py        # Weak first-principles interval construction  for industrial case in paper
├── IndustrialCase_SFPI.py        # Strong first-principles interval construction  for industrial case in paper
├── ToyDataSet.py  # Synthetic dataset and toy interval generation
├── Run.py         # Demo script
└── README.md

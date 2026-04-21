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
- `WFPI.py` – weak first-principles interval construction  
- `SFPI.py` – strong first-principles interval construction  
- `ToyDataSet.py` – synthetic dataset generation and toy interval construction  
- `Run.py` – a runnable demo script for training, evaluation, and visualization

---

## Repository Structure

```text
.
├── Method.py      # Semi-supervised temporal model
├── WFPI.py        # Weak first-principles interval construction
├── SFPI.py        # Strong first-principles interval construction
├── ToyDataSet.py  # Synthetic dataset and toy interval generation
├── Run.py         # Demo script
└── README.md

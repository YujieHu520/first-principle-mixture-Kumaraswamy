# Semi-Supervised Temporal Modeling with First-Principles Intervals and a Mixture of Kumaraswamy Distributions

This repository provides a public re-implementation of the method described in our paper on semi-supervised temporal modeling for sparse-sampling industrial processes. The method integrates first-principles interval construction with a neural network that outputs a mixture of truncated Kumaraswamy distributions.

---

## Overview

In many industrial processes, labels are sampled much more sparsely than process variables. To address this issue, this repository implements a semi-supervised temporal modeling framework with the following key ideas:

- **First-principles models** are used to construct **label intervals** rather than point predictions.
- A data-driven temporal model is trained to learn the **label location within these intervals**.
- The predictive distribution is represented as a **mixture of truncated Kumaraswamy distributions**.
- Both **labeled** and **unlabeled** samples are used during training.

The repository currently contains four main Python files:

- `Method.py` – the semi-supervised neural model and training procedure  
- `WFPI.py` – weak first-principles interval construction  
- `SFPI.py` – strong first-principles interval construction  
- `Run.py` – a runnable demo script with synthetic data generation and visualization

---

## Repository Structure

```text
.
├── Method.py   # Semi-supervised temporal model
├── WFPI.py     # Weak first-principles interval construction
├── SFPI.py     # Strong first-principles interval construction
├── Run.py      # Demo script
└── README.md

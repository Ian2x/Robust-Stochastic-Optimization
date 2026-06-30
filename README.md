# Robust Stochastic Optimization: RGD vs. SEVER

A study of how gradient-based optimizers behave when a fraction of the training data is **adversarially corrupted** — comparing **Robust Gradient Descent (RGD)** against **SEVER**, plus a hybrid of the two, across a sweep of contamination levels.

Course research for CPSC 561 (Yale, Spring 2025).

## What's here

| File | Purpose |
|---|---|
| `robust_stochastic_optimization.py` | The full experiment: data generation, ε-contamination, the RGD / SEVER / hybrid estimators, and the evaluation sweep. |
| `config.yaml` | All hyper-parameters — the outlier-fraction grid (`eps_grid`), dataset sizes, and estimator settings. |
| `results.txt` | Saved metrics from a run. |

## Background

A small fraction of adversarial outliers can break standard estimators like ordinary least squares. Two algorithms address this:

- **RGD** — *Robust estimation via robust gradient estimation* (Prasad, Suggala, Balakrishnan & Ravikumar, 2018): replace the gradient with a robust mean estimate at each optimization step.
- **SEVER** — *SEVER: A robust meta-algorithm for stochastic optimization* (Diakonikolas, Kamath, Kane, Li, Steinhardt & Stewart, 2019): iteratively filter out points whose gradients align with the top singular direction of the gradient covariance.

This repo implements both (and a hybrid), runs them on contaminated regression data across an ε-grid from 0 to 0.5, and compares how well each recovers the true parameters.

## Running it

```bash
pip install numpy pandas scipy scikit-learn matplotlib pyyaml
python robust_stochastic_optimization.py
```

Edit `config.yaml` to change the contamination grid, dataset size, or estimator settings.

"""
Comparison between SEVER and Robust Gradient Descent (RGD) on simulated data.

By:
    Ian Wang
    ChatGPT

Course:
    CPSC 561 — Yale University, Spring 2025

References:
    Diakonikolas, I., Kamath, G., Kane, D., Li, J., Steinhardt, J., & Stewart, A. (2019).
        SEVER: A robust meta-algorithm for stochastic optimization. In
        International Conference on Machine Learning (pp. 1596-1606). PMLR.

    Prasad, A., Suggala, A. S., Balakrishnan, S., & Ravikumar, P. (2018).
        Robust estimation via robust gradient estimation. CoRR, abs/1802.06485.
"""

from __future__ import annotations

import csv
import math
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng, Generator
from scipy.linalg import svd
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.metrics import mean_squared_error

##############################################################################
# 1.  Constants / Setup                                                      #
##############################################################################

OUT_DIR   = Path(".")
CSV_MSE   = OUT_DIR / "mse_results.csv"
CSV_PERR  = OUT_DIR / "param_error_results.csv"
CSV_TIME  = OUT_DIR / "runtime_results.csv"

def _init_csv(path: Path):
    path.write_text("eps,method,value\n", encoding="utf-8")

for p in [CSV_MSE, CSV_PERR, CSV_TIME]:
    _init_csv(p)

CONFIG_FILE = "config.yaml"

def load_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

##############################################################################
# 2.  Data generators                                                        #
##############################################################################

# Data generation method from [PSBR18]: Draws corrupt samples normal(0, p^2)
def generate_psbr_dataset(
    n_train: int, n_test: int, p: int, eps: float, noise_var: float, rng: Generator
):
    theta_star = np.ones(p)
    n_total = min(n_train, int(10 * p / eps**2))  if eps > 0 else n_train
    n_clean = int((1 - eps) * n_total)
    n_bad = n_total - n_clean

    X_clean = rng.normal(size=(n_clean, p))
    y_clean = X_clean @ theta_star + rng.normal(scale=math.sqrt(noise_var), size=n_clean)

    X_bad = rng.normal(scale=p, size=(n_bad, p))
    y_bad = np.zeros(n_bad)

    X_train = np.vstack([X_clean, X_bad])
    y_train = np.concatenate([y_clean, y_bad])
    is_out = np.zeros(n_total, bool)
    is_out[n_clean:] = True

    X_test = rng.normal(size=(n_test, p))
    y_test = X_test @ theta_star + rng.normal(scale=math.sqrt(noise_var), size=n_test)
    return X_train, y_train, X_test, y_test, theta_star, is_out

# Data generation method from [DKK+19b]: Creates corrupt samples that are aligned with regression direction (add_dk_outliers)
def generate_dkk_dataset(
    n_train: int, n_test: int, d: int, noise_std: float, rng: Generator
):
    w_star = rng.normal(size=d) / np.sqrt(d)
    X_train = rng.normal(size=(n_train, d))
    X_test = rng.normal(size=(n_test, d))
    y_train = X_train @ w_star + noise_std * rng.normal(size=n_train)
    y_test = X_test @ w_star + noise_std * rng.normal(size=n_test)
    return X_train, y_train, X_test, y_test, w_star

def add_dkk_outliers(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    alpha: float,
    beta: float,
    rng: Generator,
    noise_std: float = 1e-2,
):
    n_clean = X.shape[0]
    n_bad = int(math.ceil(epsilon * n_clean / (1 - epsilon)))
    if n_bad == 0:
        return X, y, np.zeros(n_clean, bool)

    x_ref = (y @ X) / (alpha * n_bad)
    X_bad = x_ref + noise_std * rng.normal(size=(n_bad, X.shape[1]))
    y_bad = -beta * np.ones(n_bad)

    X_aug = np.vstack([X, X_bad])
    y_aug = np.concatenate([y, y_bad])
    is_out = np.concatenate([np.zeros(n_clean, bool), np.ones(n_bad, bool)])
    return X_aug, y_aug, is_out

##############################################################################
# 3.  SEVER components                                                       #
##############################################################################

def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> Ridge:
    mdl = Ridge(alpha=lam, fit_intercept=False)
    mdl.fit(X, y)
    return mdl

# l2 distance of covariates from mean
def score_l2(X, y, model: Ridge, m=None):
    m = m if m is not None else X.mean(axis=0)
    return np.linalg.norm(X - m, axis=1)

# loss as measured by base learner
def score_loss(X, y, model: Ridge, **_):
    residuals = y - model.predict(X)
    return residuals**2

# l2 distance of gradients from mean
def score_gradient_centered(X, y, model: Ridge, **_):
    res = model.predict(X) - y
    grads = X * res[:, None]
    centred = grads - grads.mean(axis=0, keepdims=True)
    return np.einsum("ij,ij->i", centred, centred)

# See Algorithm 2 of [DKK+19b] for details
def score_sever(X, y, model: Ridge, **_):
    res = model.predict(X) - y
    grads = X * res[:, None]
    centred = grads - grads.mean(axis=0, keepdims=True)
    _, _, vt = svd(centred, full_matrices=False)
    v = vt[0]
    return (centred @ v) ** 2

##############################################################################
# 4.  RGD components                                                         #
##############################################################################

# See Algorithm 4 of [PSBR18] for details
# This version is simplified version in that it doesn't factor in delta portion
def _huber_outlier_truncation(S, eps, delta, rng=None):
    n, p = S.shape
    if eps <= 0 or n == 0:
        return S.copy()
    rng = rng or default_rng()
    if p == 1:
        vals = np.sort(S[:, 0])
        k = max(1, min(int((1 - eps) * n), n))
        lo, hi = vals[(n - k) // 2], vals[(n + k) // 2 - 1]
        return S[(S[:, 0] >= lo) & (S[:, 0] <= hi)]
    centre = np.median(S, axis=0)
    radii = np.linalg.norm(S - centre, axis=1)
    k = max(1, min(int((1 - eps) * n), n - 1))
    r_thr = np.partition(radii, k)[k]
    return S[radii <= r_thr]

# See Algorithm 2 of [PSBR18] for details
def _recursive_huber_grad(S, eps, delta, rng=None):
    n, p = S.shape
    rng = rng or default_rng()
    S_e = _huber_outlier_truncation(S, eps, delta, rng)
    if p == 1 or S_e.shape[0] == 0:
        return np.array([S_e.mean()]) if p == 1 else np.zeros(p)
    mu_c = S_e.mean(axis=0)
    centred = S_e - mu_c
    _, _, vt = svd(centred, full_matrices=False)
    r = max(1, p // 2)
    V, W = vt[:r].T, vt[r:].T
    mu_V = V @ _recursive_huber_grad(centred @ V, eps, delta, rng)
    mu_W = W @ (centred @ W).mean(axis=0)
    return mu_c + mu_V + mu_W

def huber_gradient_estimator(theta, X, y, lam, eps, delta, rng=None):
    rng = rng or default_rng()
    preds = X @ theta
    per_grad = X * (preds - y)[:, None] + lam * theta
    return _recursive_huber_grad(per_grad, eps, delta, rng)

def robust_gradient_descent(X, y, lam, eps, eta=0.5, T=50, delta=1e-4, rng=None):
    if eps <= 0:
        return fit_ridge(X, y, lam)
    rng = rng or default_rng()
    theta = np.zeros(X.shape[1])
    for _ in range(T):
        g_hat = huber_gradient_estimator(theta, X, y, lam, eps, delta, rng)
        theta -= eta * g_hat

    class _RGDModel:
        def __init__(self, coef_):
            self.coef_ = coef_

        def predict(self, X_):
            return X_ @ self.coef_

    return _RGDModel(theta)

##############################################################################
# 5.  SEVER r-round filters                                                  #
##############################################################################

# See Algorithm 1 of [DKK+19b] for details
def run_filter(X, y, lam, epsilon, score_fn, is_out, r, rng=None):
    rng = rng or default_rng()
    Xc, yc, mask_c = X.copy(), y.copy(), is_out.copy()
    k_remove = int((epsilon / 2) * len(X))
    score_trace, mask_trace = [], []
    for _ in range(r):
        mdl = fit_ridge(Xc, yc, lam)
        scores = score_fn(Xc, yc, mdl)
        score_trace.append(scores.copy())
        mask_trace.append(mask_c.copy())
        if k_remove <= 0 or k_remove >= len(scores):
            break
        idx = scores.argsort()[::-1]
        keep = np.ones(len(scores), bool)
        keep[idx[:k_remove]] = False
        Xc, yc, mask_c = Xc[keep], yc[keep], mask_c[keep]
    final = fit_ridge(Xc, yc, lam)
    return final, score_trace, mask_trace

# Novel implementation merging SEVER and RGD
# Same as run_filter but using RGD instead of Ridge
def run_filter_rgd(
    X, y, lam, epsilon, score_fn, is_out, r, rng=None, eta=0.5, T=50, delta=1e-4
):
    if epsilon <= 0:
        return fit_ridge(X, y, lam), [], []
    rng = rng or default_rng()
    Xc, yc, mask_c = X.copy(), y.copy(), is_out.copy()
    k_remove = int((epsilon / 2) * len(X))
    score_trace, mask_trace = [], []
    for _ in range(r):
        mdl = robust_gradient_descent(Xc, yc, lam, epsilon, eta, T, delta, rng)
        scores = score_fn(Xc, yc, mdl)
        score_trace.append(scores.copy())
        mask_trace.append(mask_c.copy())
        if k_remove <= 0 or k_remove >= len(scores):
            break
        idx = scores.argsort()[::-1]
        keep = np.ones(len(scores), bool)
        keep[idx[:k_remove]] = False
        Xc, yc, mask_c = Xc[keep], yc[keep], mask_c[keep]
    final = robust_gradient_descent(Xc, yc, lam, epsilon, eta, T, delta, rng)
    return final, score_trace, mask_trace

##############################################################################
# 6.  Experiment loop                                                        #
##############################################################################

def run_experiment(cfg: dict, methods: List[str]):
    eps_grid  = cfg["eps_grid"]
    res_mse   = {m: [] for m in methods}
    res_perr  = {m: [] for m in methods}
    snapshots: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    rng_master = default_rng(cfg["seed"])
    for eps in eps_grid:
        print(f"\n=== eps = {eps:.3f} ===")
        errs      = {m: [] for m in methods}
        perr_vals = {m: [] for m in methods}
        times_eps = {m: [] for m in methods}

        # Run multiple repetitions for each epsilon for robustness
        for rep in range(cfg["repeats"]):
            rng = default_rng(rng_master.integers(1 << 32))

            # Data generation
            if cfg["alt_setup"]:
                Xtr, ytr, Xte, yte, w_true, is_out = generate_psbr_dataset(
                    cfg["n_train"], cfg["n_test"], cfg["d"],
                    eps, cfg["noise_std"]**2, rng
                )
            else:
                Xtr, ytr, Xte, yte, w_true = generate_dkk_dataset(
                    cfg["n_train"], cfg["n_test"], cfg["d"],
                    cfg["noise_std"], rng
                )
                Xtr, ytr, is_out = add_dkk_outliers(
                    Xtr, ytr, eps,
                    cfg["alpha_out"], cfg["beta_out"], rng
                )

            # --- uncorrupted baseline ---
            start = time.time()
            mdl = fit_ridge(Xtr, ytr, cfg["lam"])
            times_eps["uncorrupted"].append(time.time() - start)
            errs["uncorrupted"].append(mean_squared_error(yte, mdl.predict(Xte)))
            perr_vals["uncorrupted"].append(np.linalg.norm(mdl.coef_ - w_true))

            # --- l2 filter ---
            start = time.time()
            mdl_l2, tr_l2, ms_l2 = run_filter(
                Xtr, ytr, cfg["lam"],
                eps, score_l2, is_out,
                cfg["filter_rounds"], rng
            )
            times_eps["l2"].append(time.time() - start)
            errs["l2"].append(mean_squared_error(yte, mdl_l2.predict(Xte)))
            perr_vals["l2"].append(np.linalg.norm(mdl_l2.coef_ - w_true))

            # --- loss filter ---
            start = time.time()
            mdl_ls, tr_ls, ms_ls = run_filter(
                Xtr, ytr, cfg["lam"],
                eps, score_loss, is_out,
                cfg["filter_rounds"], rng
            )
            times_eps["loss"].append(time.time() - start)
            errs["loss"].append(mean_squared_error(yte, mdl_ls.predict(Xte)))
            perr_vals["loss"].append(np.linalg.norm(mdl_ls.coef_ - w_true))

            # --- gradientCentered filter ---
            start = time.time()
            mdl_gc, tr_gc, ms_gc = run_filter(
                Xtr, ytr, cfg["lam"],
                eps, score_gradient_centered, is_out,
                cfg["filter_rounds"], rng
            )
            times_eps["gradientCentered"].append(time.time() - start)
            errs["gradientCentered"].append(mean_squared_error(yte, mdl_gc.predict(Xte)))
            perr_vals["gradientCentered"].append(np.linalg.norm(mdl_gc.coef_ - w_true))

            # --- RANSAC baseline ---
            start = time.time()
            rr = RANSACRegressor(
                Ridge(alpha=cfg["lam"], fit_intercept=False),
                min_samples=0.5,
                random_state=rng.integers(1 << 32),
            )
            rr.fit(Xtr, ytr)
            times_eps["RANSAC"].append(time.time() - start)
            errs["RANSAC"].append(mean_squared_error(yte, rr.predict(Xte)))
            perr_vals["RANSAC"].append(np.linalg.norm(rr.estimator_.coef_ - w_true))

            # --- Sever filter ---
            start = time.time()
            mdl_sv, tr_sv, ms_sv = run_filter(
                Xtr, ytr, cfg["lam"],
                eps, score_sever, is_out,
                cfg["filter_rounds"], rng
            )
            times_eps["Sever"].append(time.time() - start)
            errs["Sever"].append(mean_squared_error(yte, mdl_sv.predict(Xte)))
            perr_vals["Sever"].append(np.linalg.norm(mdl_sv.coef_ - w_true))

            # --- plain RGD ---
            start = time.time()
            mdl_rgd = robust_gradient_descent(
                Xtr, ytr, cfg["lam"], eps,
                cfg["rgd"]["eta"], cfg["rgd"]["T"],
                cfg["rgd"]["delta"], rng
            )
            times_eps["RGD"].append(time.time() - start)
            errs["RGD"].append(mean_squared_error(yte, mdl_rgd.predict(Xte)))
            perr_vals["RGD"].append(np.linalg.norm(mdl_rgd.coef_ - w_true))

            # --- SeverRGD filter ---
            start = time.time()
            mdl_srgd, tr_srgd, ms_srgd = run_filter_rgd(
                Xtr, ytr, cfg["lam"],
                eps, score_sever, is_out,
                cfg["filter_rounds"],
                rng, cfg["rgd"]["eta"],
                cfg["rgd"]["T"], cfg["rgd"]["delta"]
            )
            times_eps["SeverRGD"].append(time.time() - start)
            errs["SeverRGD"].append(mean_squared_error(yte, mdl_srgd.predict(Xte)))
            perr_vals["SeverRGD"].append(np.linalg.norm(mdl_srgd.coef_ - w_true))

            # Snapshot at eps=0.1 (for generating histograms of SEVER rounds)
            if eps == 0.1 and rep == 0:
                snapshots["l2"]              = (tr_l2,  ms_l2)
                snapshots["loss"]            = (tr_ls,  ms_ls)
                snapshots["gradientCentered"]= (tr_gc,  ms_gc)
                snapshots["Sever"]           = (tr_sv,  ms_sv)
                snapshots["SeverRGD"]        = (tr_srgd, ms_srgd)

        # Write medians to CSV
        with open(CSV_MSE,  "a", newline="") as f_mse, \
             open(CSV_PERR, "a", newline="") as f_perr, \
             open(CSV_TIME, "a", newline="") as f_time:

            w_mse  = csv.writer(f_mse)
            w_perr = csv.writer(f_perr)
            w_time = csv.writer(f_time)

            for m in methods:
                m_mse  = float(np.median(errs[m]))
                m_perr = float(np.median(perr_vals[m]))
                m_time = float(np.median(times_eps[m]))
                w_mse .writerow([eps, m, m_mse])
                w_perr.writerow([eps, m, m_perr])
                w_time.writerow([eps, m, m_time])
                res_mse [m].append(m_mse)
                res_perr[m].append(m_perr)

    return eps_grid, res_mse, res_perr, snapshots


##############################################################################
# 7.  Plot helpers                                                           #
##############################################################################

def plot_from_csv(csv_path: Path, ylabel: str, title: str, out_path: Path):
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="eps", columns="method", values="value").sort_index()
    plt.figure(figsize=(6, 4))
    for method in pivot.columns:
        plt.plot(pivot.index, pivot[method], marker="o", label=method)
    plt.xlabel("Outlier fraction ε")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_histograms_rounds(
    score_list: List[np.ndarray],
    mask_list: List[np.ndarray],
    title: str,
    out_file: Path,
):
    R = len(score_list)
    fig, axes = plt.subplots(1, R, figsize=(4 * R, 3), sharey=True)
    if R == 1:
        axes = [axes]

    for r, (ax, scores, mask) in enumerate(zip(axes, score_list, mask_list)):
        bins = np.linspace(scores.min(), scores.max(), 60)
        ax.hist(
            scores[~mask],
            bins=bins,
            color="tab:blue",
            alpha=0.5,
            edgecolor="black",
        )
        if mask.any():
            ax.hist(
                scores[mask],
                bins=bins,
                color="tab:red",
                alpha=0.8,
                edgecolor="black",
            )
        ax.set_yscale("log")
        ax.set_title(f"{title} – round {r}")
        ax.set_xlabel("score")

    axes[0].set_ylabel("frequency (log)")
    axes[-1].legend(["clean", "outliers"], loc="upper right")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

def plot_runtime_table(eps: float, csv_time: Path, out_path: Path):
    df = pd.read_csv(csv_time)
    df0 = df[df["eps"] == eps].sort_values("method")
    methods = df0["method"].tolist()
    times   = df0["value"].tolist()

    cell_text = [[f"{t:.4f}"] for t in times]
    col_labels = ["median runtime (s)"]

    fig, ax = plt.subplots(figsize=(4, 0.5 + 0.3 * len(methods)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=methods,
        colLabels=col_labels,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.title(f"Runtimes at ε={eps}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

##############################################################################
# 8.  Main                                                                   #
##############################################################################

def main():
    cfg = load_config()
    methods = [
        "uncorrupted", "l2", "loss", "gradientCentered",
        "RANSAC", "Sever", "RGD", "SeverRGD",
    ]

    # --- Run experiment ---
    eps_grid, res_mse, res_perr, snapshots = run_experiment(cfg, methods)

    # --- Plot results ---
    plot_from_csv(
        CSV_MSE,
        "Test MSE",
        "ε vs test error",
        OUT_DIR / "fig_epsilon_vs_error.png",
    )

    plot_from_csv(
        CSV_PERR,
        "‖ŵ - w★‖₂",
        "ε vs parameter error",
        OUT_DIR / "fig_epsilon_vs_param_error.png",
    )

    for name in ["l2", "loss", "gradientCentered", "Sever", "SeverRGD"]:
        if name in snapshots:
            sc, mk = snapshots[name]
            plot_histograms_rounds(
                score_list=sc,
                mask_list=mk,
                title=name,
                out_file=OUT_DIR / f"{name.lower()}_rounds.png",
            )

    plot_runtime_table(
        0.1,
        CSV_TIME,
        OUT_DIR / "runtime_table_eps0.1.png",
    )

    # --- Print fit result tables ---
    name_w = max(map(len, methods)) + 2
    print("\nMedian Test MSE:")
    print("eps".ljust(6) + "".join(m.ljust(name_w) for m in methods))
    for i, e in enumerate(eps_grid):
        print(f"{e:0.3f}".ljust(6) +
              "".join(f"{res_mse[m][i]:10.4f}".ljust(name_w) for m in methods))

    print("\nMedian Parameter Error (‖ŵ - w★‖₂):")
    print("eps".ljust(6) + "".join(m.ljust(name_w) for m in methods))
    for i, e in enumerate(eps_grid):
        print(f"{e:0.3f}".ljust(6) +
              "".join(f"{res_perr[m][i]:10.4f}".ljust(name_w) for m in methods))


if __name__ == "__main__":
    main()

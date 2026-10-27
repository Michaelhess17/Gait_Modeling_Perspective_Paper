#!/usr/bin/env python
"""SSMLearnPy gait prediction figure.

Uses the best setting found in `SSM_hyperparam_scout.py`:
30-sample delay embedding, 2D linear chart, cubic geometry, and a degree-9
reduced map. Also includes a Van der Pol sanity check showing that the same
SSMLearnPy machinery can model a nonlinear oscillator when the assumptions are
well matched.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "cache/matplotlib").resolve()))
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from ssmlearnpy import SSMLearn

np.seterr(all="ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("SSMLearn").setLevel(logging.ERROR)
logging.getLogger("ridge_regression").setLevel(logging.ERROR)

DATA_PATH = Path("data/all_human_data.npy")
META_PATH = Path("data/all_human_data_metadata.csv")
PROMISING_PATH = (ROOT / "cache/ssm_promising_full.json")
CACHE_PATH = (ROOT / "cache/fig7_ssmlearnpy_primary.npz")
SUMMARY_PATH = (ROOT / "cache/fig7_ssmlearnpy_primary_summary.json")
FIG_PNG = Path("figures/Figure_7_SSM.png")
FIG_SVG = Path("figures/Figure_7_SSM.svg")

FS = 100.0
TRAIN_STEPS = 1000
TEST_STEPS = 500
DELAY_LAGS = 30
SSM_DIM = 2
GEOMETRY_DEGREE = 3
DYNAMICS_DEGREE = 9
ALPHA = 1e-4
SCRIPT_VERSION = 4

GROUP_COLORS = {"AB": "#4575b4", "HF": "#fdae61", "LF": "#d73027"}


def delay_embed(x: np.ndarray, lags: int) -> np.ndarray:
    if lags == 0:
        return x.T
    return np.vstack([x[i : x.shape[0] - lags + i].T for i in range(lags + 1)])


def prepare_trial(x_trial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_train = x_trial[:TRAIN_STEPS]
    x_test = x_trial[TRAIN_STEPS - DELAY_LAGS : TRAIN_STEPS + TEST_STEPS]
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return delay_embed((x_train - mean) / scale, DELAY_LAGS), delay_embed((x_test - mean) / scale, DELAY_LAGS)


def fit_predict_gait(x_trial: np.ndarray) -> dict:
    h_train, h_test = prepare_trial(x_trial)
    t_train = np.arange(h_train.shape[1]) / FS
    t_test = np.arange(h_test.shape[1]) / FS

    model = SSMLearn(
        t=[t_train],
        x=[h_train],
        derive_embdedding=False,
        ssm_dim=SSM_DIM,
        dynamics_type="map",
        dynamics_structure="generic",
        error_metric="NTE",
    )
    model.get_reduced_coordinates("linearchart")
    model.get_parametrization(
        poly_degree=GEOMETRY_DEGREE,
        alpha=ALPHA,
        cv=0,
        do_scaling=True,
        fit_intercept=False,
    )
    model.get_reduced_dynamics(
        poly_degree=DYNAMICS_DEGREE,
        alpha=ALPHA,
        cv=0,
        do_scaling=True,
        fit_intercept=False,
    )
    pred = model.predict(t=[t_test], x=[h_test])
    y_true = h_test[:6, :]
    y_pred = pred["observables"][0][:6, : y_true.shape[1]]
    finite = np.isfinite(y_pred).all(axis=0)
    if not np.all(finite):
        y_pred[:, int(np.argmax(~finite)) :] = np.nan
    rmse_t = np.sqrt(np.nanmean((y_pred - y_true) ** 2, axis=0))
    rmse_t[~np.isfinite(rmse_t)] = np.nan
    return {"rmse_t": rmse_t, "true": y_true, "pred": y_pred}


def simulate_van_der_pol(mu: float = 2.0, dt: float = 0.02, t_end: float = 40.0):
    def rhs(_t, u):
        x, v = u
        return [v, mu * (1.0 - x * x) * v - x]

    t_eval = np.arange(0.0, t_end + 0.5 * dt, dt)
    sol = solve_ivp(rhs, (0.0, t_end), [-3.0, 0.0], t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y


def oscillator_sanity() -> dict:
    t, x = simulate_van_der_pol()
    train_t = t[:1200]
    train_x = x[:, :1200]
    test_t = t[1200:1800] - t[1200]
    test_x = x[:, 1200:1800]
    mean = train_x.mean(axis=1, keepdims=True)
    scale = train_x.std(axis=1, keepdims=True)
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale

    model = SSMLearn(
        t=[train_t],
        x=[train_z],
        reduced_coordinates=[train_z],
        derive_embdedding=False,
        ssm_dim=2,
        dynamics_type="map",
        dynamics_structure="generic",
        error_metric="NTE",
    )
    model.get_parametrization(poly_degree=5, alpha=1e-6, cv=0, do_scaling=True, fit_intercept=False)
    model.get_reduced_dynamics(poly_degree=5, alpha=1e-6, cv=0, do_scaling=True, fit_intercept=False)
    pred = model.predict(t=[test_t], x=[test_z], x_reduced=[test_z])
    y_pred = pred["observables"][0][:2, :]
    rmse_t = np.sqrt(np.mean((y_pred - test_z) ** 2, axis=0))
    return {"t": test_t, "true": test_z, "pred": y_pred, "rmse_t": rmse_t}


def comparison_rows() -> list[dict]:
    if PROMISING_PATH.exists():
        rows = json.loads(PROMISING_PATH.read_text())
        return rows
    return []


def run_all() -> dict:
    data = np.load(DATA_PATH)
    meta = pd.read_csv(META_PATH)
    groups = meta["lf_or_hf"].to_numpy()
    rmse = np.full((data.shape[0], TEST_STEPS), np.nan)
    true_examples: dict[str, np.ndarray] = {}
    pred_examples: dict[str, np.ndarray] = {}
    trial_payload = {}

    for i in range(data.shape[0]):
        if i % 25 == 0:
            print(f"trial {i}/{data.shape[0]}", flush=True)
        try:
            result = fit_predict_gait(data[i])
            n = min(TEST_STEPS, result["rmse_t"].shape[0])
            rmse[i, :n] = result["rmse_t"][:n]
            trial_payload[i] = result
        except Exception as exc:
            print(f"trial {i} failed: {type(exc).__name__}: {exc}", flush=True)

    example_indices = {}
    for group in ["AB", "HF", "LF"]:
        idx = np.where((groups == group) & np.isfinite(rmse[:, 299]))[0]
        if len(idx):
            med = np.nanmedian(rmse[idx, 299])
            chosen = int(idx[np.argmin(np.abs(rmse[idx, 299] - med))])
            example_indices[group] = chosen
            true_examples[group] = trial_payload[chosen]["true"]
            pred_examples[group] = trial_payload[chosen]["pred"]

    osc = oscillator_sanity()
    np.savez_compressed(
        CACHE_PATH,
        rmse=rmse,
        groups=groups,
        example_indices=np.array([example_indices.get(g, -1) for g in ["AB", "HF", "LF"]]),
        osc_t=osc["t"],
        osc_true=osc["true"],
        osc_pred=osc["pred"],
        osc_rmse_t=osc["rmse_t"],
        **{f"example_{g}_true": v for g, v in true_examples.items()},
        **{f"example_{g}_pred": v for g, v in pred_examples.items()},
    )
    return load_results()


def load_results() -> dict:
    arr = np.load(CACHE_PATH, allow_pickle=True)
    examples = {}
    for group in ["AB", "HF", "LF"]:
        tk, pk = f"example_{group}_true", f"example_{group}_pred"
        if tk in arr and pk in arr:
            examples[group] = {"true": arr[tk], "pred": arr[pk]}
    return {
        "rmse": arr["rmse"],
        "groups": arr["groups"],
        "examples": examples,
        "osc": {
            "t": arr["osc_t"],
            "true": arr["osc_true"],
            "pred": arr["osc_pred"],
            "rmse_t": arr["osc_rmse_t"],
        },
    }


def summarize(results: dict) -> dict:
    rmse = results["rmse"]
    summary = {
        "settings": {
            "script_version": SCRIPT_VERSION,
            "delay_lags": DELAY_LAGS,
            "ssm_dim": SSM_DIM,
            "geometry_degree": GEOMETRY_DEGREE,
            "dynamics_degree": DYNAMICS_DEGREE,
            "alpha": ALPHA,
        },
        "n_trials": int(rmse.shape[0]),
        "finite_1s": int(np.isfinite(rmse[:, 99]).sum()),
        "finite_3s": int(np.isfinite(rmse[:, 299]).sum()),
        "finite_5s": int(np.isfinite(rmse[:, 499]).sum()),
        "median_rmse_1s": float(np.nanmedian(rmse[:, 99])),
        "median_rmse_3s": float(np.nanmedian(rmse[:, 299])),
        "median_rmse_5s": float(np.nanmedian(rmse[:, 499])),
        "groups": {},
        "oscillator_median_rmse": float(np.nanmedian(results["osc"]["rmse_t"])),
    }
    for group in ["AB", "HF", "LF"]:
        mask = results["groups"] == group
        summary["groups"][group] = {
            "n": int(mask.sum()),
            "finite_3s": int(np.isfinite(rmse[mask, 299]).sum()),
            "median_rmse_3s": float(np.nanmedian(rmse[mask, 299])),
        }
    return summary


def plot_results(results: dict, summary: dict) -> None:
    t = np.arange(TEST_STEPS) / FS
    fig = plt.figure(figsize=(12, 8.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.0], hspace=0.58, wspace=0.34)

    ax0 = fig.add_subplot(gs[0, :2])
    for group in ["AB", "HF", "LF"]:
        curves = results["rmse"][results["groups"] == group]
        ax0.plot(t, np.nanmedian(curves, axis=0), color=GROUP_COLORS[group], lw=2.0, label=group)
        ax0.fill_between(
            t,
            np.nanpercentile(curves, 25, axis=0),
            np.nanpercentile(curves, 75, axis=0),
            color=GROUP_COLORS[group],
            alpha=0.18,
            linewidth=0,
        )
    ax0.axhline(1.0, color="0.25", lw=1, ls="--")
    ax0.set_title("(A) Best SSMLearnPy gait forecast")
    ax0.set_xlabel("Held-out prediction time (s)")
    ax0.set_ylabel("Joint-angle NRMSE")
    ax0.set_ylim(bottom=0)
    ax0.legend(frameon=False, ncol=3, loc="upper left")

    ax1 = fig.add_subplot(gs[0, 2])
    rows = comparison_rows()
    labels, vals, finite = [], [], []
    for row in rows[:4]:
        cfg = row["config"]
        labels.append(cfg["name"].replace("map_", "").replace("_", "\n"))
        vals.append(row["median_rmse_3s"] if row["median_rmse_3s"] is not None else np.nan)
        finite.append(row["finite_3s"] / row["n"])
    x = np.arange(len(vals))
    ax1.bar(x, vals, color="#91bfdb", edgecolor="0.3")
    ax1.axhline(1.0, color="0.25", lw=1, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("Median 3 s NRMSE")
    ax1.set_title("(B) Hyperparameter scout")
    for xi, fr in zip(x, finite):
        ax1.text(xi, 0.03, f"{fr:.0%}", ha="center", va="bottom", fontsize=8, rotation=90)

    for j, group in enumerate(["AB", "HF", "LF"]):
        ax = fig.add_subplot(gs[1, j])
        ex = results["examples"].get(group)
        if ex:
            true = ex["true"][1, :]
            pred = ex["pred"][1, :]
            ax.plot(t[: len(true)], true, color="black", lw=1.35, label="Observed")
            ax.plot(t[: len(pred)], pred, color=GROUP_COLORS[group], lw=1.35, label="SSM pred.")
        ax.axvspan(0, 0.25, color="0.92", lw=0)
        ax.set_title(f"({chr(ord('C') + j)}) {group} representative")
        ax.set_xlabel("Held-out time (s)")
        if j == 0:
            ax.set_ylabel("Standardized knee angle")
            ax.legend(frameon=False, fontsize=8, loc="upper right")

    osc = results["osc"]
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(osc["t"], osc["true"][0], color="black", lw=1.5, label="Observed")
    ax5.plot(osc["t"], osc["pred"][0], color="#2ca25f", lw=1.5, label="SSM pred.")
    ax5.set_title("(F) Van der Pol sanity check")
    ax5.set_xlabel("Held-out time (s)")
    ax5.set_ylabel("Standardized x")
    ax5.legend(frameon=False, fontsize=8)

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(osc["t"], osc["rmse_t"], color="#2ca25f", lw=1.8)
    ax6.axhline(1.0, color="0.25", lw=1, ls="--")
    ax6.set_title("(G) Oscillator error")
    ax6.set_xlabel("Held-out time (s)")
    ax6.set_ylabel("State NRMSE")
    ax6.set_ylim(bottom=0)

    fig.suptitle(
        "SSMLearnPy can forecast gait with tuned delay-map settings, but remains an empirical predictor",
        y=0.99,
        fontsize=13,
    )
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_SVG, bbox_inches="tight")


def main() -> None:
    (ROOT / "cache").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    use_cache = False
    if CACHE_PATH.exists() and SUMMARY_PATH.exists():
        old = json.loads(SUMMARY_PATH.read_text())
        use_cache = old.get("settings", {}).get("script_version") == SCRIPT_VERSION
    results = load_results() if use_cache else run_all()
    summary = summarize(results)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    plot_results(results, summary)
    print(json.dumps(summary, indent=2))
    print(f"saved {FIG_PNG}")
    print(f"saved {FIG_SVG}")


if __name__ == "__main__":
    main()

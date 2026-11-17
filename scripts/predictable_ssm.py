#!/usr/bin/env python
"""SSMLearnPy on the predictable-response test (predictable_more.jl exports the data and scores the forecasts).
Per configuration: settings from maus_ssm.py's whole-signal grid (delay window up to a stride, manifold dimension
2-16, map degree, ridge), chosen by the same criterion as the other models (mean squared error of 2-step-long
rollouts, 100 samples, from the 20 validation histories); then free runs of 300 samples from every test history
(unclipped; non-finite values left as NaN, which predictable_more.jl counts as diverged), and 30-stride free runs from
the first test start of each held-out walk (R2).

    .pixi/envs/default/bin/python scripts/predictable_ssm.py   -> cache/predictable/ssm_io/forecast_N.npz
"""
from __future__ import annotations

import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import maus_ssm as m

warnings.filterwarnings("ignore")
DIR = Path(__file__).resolve().parent.parent / "cache/predictable/ssm_io"


def rollout(M, H, n):
    """Free runs of n samples from histories H (hist x d x starts); returns n x d x starts."""
    model, mu, lags = M
    d = H.shape[1]
    H0 = np.stack([(H[-(lags + 1):, :, i] - mu).reshape(-1) for i in range(H.shape[2])], axis=1)
    z = model.encoder.predict(H0)
    out = np.full((n, d, H.shape[2]), np.nan)
    with np.errstate(all="ignore"):
        for s in range(n):
            z = model.reduced_dynamics.predict(np.nan_to_num(z.T, nan=1e6, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)).T
            P = model.decoder.predict(z.T)[:, -d:] + mu
            P[~np.isfinite(P)] = np.nan
            P[np.abs(P) > 1e6] = np.nan
            out[s] = P.T
    return out


def config(i: int):
    Z = np.load(DIR / f"config_{i}.npz")
    Ytr, VH, VF, TH, hor = Z["Ytr"], Z["VH"], Z["VF"], Z["TH"], int(Z["hor"])
    rows = slice(0, Ytr.shape[0])
    best, sbest = np.inf, None
    for s in m.SETTINGS_WHOLE:
        try:
            F = rollout(m.fit(Ytr, rows, s), VH, VF.shape[0])
            e = np.mean((F - VF) ** 2)
            if np.isfinite(e) and e < best:
                best, sbest = e, s
        except Exception:
            continue
    M = m.fit(Ytr, rows, sbest)
    F = rollout(M, TH, hor)
    FL = rollout(M, TH[:, :, Z["first"] - 1], int(Z["nrunl"]))          # 30-stride free runs (R2)
    np.savez(DIR / f"forecast_{i}.npz", F=F, FL=FL, setting=np.array(sbest, dtype=float))
    return i, sbest, best


def main() -> None:
    n = len(list(DIR.glob("config_*.npz")))
    with Pool(n) as pool:
        for i, s, e in pool.map(config, range(1, n + 1)):
            print(f"config {i}: setting {s}, validation error {e:.4f}")


if __name__ == "__main__":
    main()

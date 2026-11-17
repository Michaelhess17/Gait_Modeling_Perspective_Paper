#!/usr/bin/env python
"""R2 for SSMLearnPy on the running whole signal (maus_whole_r2.jl classifies the runs): each trial refitted on its 150
training strides with the settings maus_ssm.py chose (cache/maus/ssm.npz), then run freely for 30 strides (3000
samples) from the end of the training strides, unclipped, in real time and in stride-normalised time.

    .pixi/envs/default/bin/python scripts/maus_ssm_whole_r2.py   -> cache/maus/ssm_whole_runs.npz
"""
from __future__ import annotations

import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import maus_ssm as m

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
NRUN = 3000
SET = np.load(ROOT / "cache/maus/ssm.npz")["whole_setting"]           # trials × (rt, sn) × (lags, dim, degree, ridge)


def run(M, Y, t, n):
    model, mu, lags = M
    p = Y.shape[1]
    z = model.encoder.predict((Y[t - lags: t + 1] - mu).reshape(-1, 1))
    out = np.full((n, p), np.nan)
    with np.errstate(all="ignore"):
        for s in range(n):
            z = model.reduced_dynamics.predict(np.nan_to_num(z.T, nan=1e6, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)).T
            P = model.decoder.predict(z.T)[0, -p:] + mu
            if not np.isfinite(P).all() or np.abs(P).max() > 1e6:
                break
            out[s] = P
    return out


def trial(i: int):
    Z = np.load(ROOT / f"cache/maus/ssm_input/trial_{i + 1:02d}.npz")
    res = []
    for j, rung in enumerate(("rt", "sn")):
        Y, b = Z["Y" + rung], Z["bidx_" + rung] - 1
        s = tuple(SET[i, j]); s = (int(s[0]), int(s[1]), int(s[2]), float(s[3]))
        rows = slice(0, b[150])
        res.append(run(m.fit(Y, rows, s), Y, b[150] - 1, NRUN))
    return np.stack(res)


def main() -> None:
    n = SET.shape[0]
    with Pool(min(n, 16)) as pool:
        R = pool.map(trial, range(n))
    np.savez(ROOT / "cache/maus/ssm_whole_runs.npz", runs=np.stack(R))           # trials × 2 × NRUN × 6
    print("saved", np.stack(R).shape)


if __name__ == "__main__":
    main()

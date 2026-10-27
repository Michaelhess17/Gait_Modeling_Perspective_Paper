#!/usr/bin/env python
"""SSMLearnPy in the self-driving benchmark (self_driving_benchmark.jl).

Reads the exact training/held-out data exported by the Julia benchmark
(cache/selfdrive/data_<system>_<regime>.npz), fits the tuned SSMLearnPy
delay-map configuration used for gait (2-D linear chart, cubic manifold,
degree-9 reduced map), runs it freely for 30 cycles from the delay vector ending
at the last training sample, and scores it exactly as the Julia benchmark does.
The forecast is read from the NEWEST block of the delay vector, so step k is the
prediction of training sample n + k.

Delay depth is about a quarter cycle (the gait configuration used 30 lags at
100 Hz). Two variants are run per cell: degree-5 and degree-9 reduced maps.

    pixi run python ssm_self_driving.py
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
from ssmlearnpy import SSMLearn

warnings.filterwarnings("ignore")
for name in ("", "SSMLearn", "ridge_regression"):
    logging.getLogger(name).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
FATES = ["ok", "decays", "wrong", "collapse", "blow-up"]   # order shared with Fig_self_driving.jl
DIR = ROOT / "cache/selfdrive"


def delay_embed(x: np.ndarray, lags: int) -> np.ndarray:
    """Columns are delay vectors, oldest block first, newest block last."""
    return np.vstack([x[i : x.shape[0] - lags + i].T for i in range(lags + 1)])


def period_samples(y: np.ndarray, lo: int = 20) -> float:
    """Same autocorrelation period estimate as the Julia benchmark."""
    x = y[:, 0] - y[:, 0].mean()
    n = len(x)
    ac = np.fft.ifft(np.abs(np.fft.fft(np.concatenate([x, np.zeros(n)]))) ** 2).real[:n]
    ac /= ac[0]
    pk = [i for i in range(lo, n // 2) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.3]
    if not pk:
        return np.nan
    i = pk[0]
    a, b, c = ac[i - 1], ac[i], ac[i + 1]
    return i + 0.5 * (a - c) / (a - 2 * b + c)


def score(fc: np.ndarray, yte: np.ndarray, ytr: np.ndarray, P: float) -> dict:
    errs = []
    for h in (1, 3, 5):
        n = min(int(round(h * P)), yte.shape[0])
        seg = fc[:n]
        errs.append(np.inf if not np.isfinite(seg).all()
                    else np.sqrt(np.mean((seg - yte[:n]) ** 2)) / np.sqrt(np.mean(yte[:n] ** 2)))
    tail = fc[-int(round(10 * P)):]
    if not np.isfinite(tail).all():
        return dict(e1=errs[0], e3=errs[1], e5=errs[2], amp=np.inf, per=np.nan, fate="blow-up")
    amp = tail.std(axis=0).mean() / ytr.std(axis=0).mean()
    per = period_samples(tail) / P if amp > 0.05 else np.nan
    rhythm = np.isfinite(per) and abs(per - 1) < 0.1
    fate = ("blow-up" if amp > 4 else "collapse" if amp < 0.25 else
            "wrong" if (not rhythm or amp > 1.33) else "ok" if amp >= 0.75 else "decays")
    return dict(e1=errs[0], e3=errs[1], e5=errs[2], amp=amp, per=per, fate=fate)


def fit_free_run(ytr: np.ndarray, lags: int, n_run: int, dyn_degree: int) -> np.ndarray:
    d = ytr.shape[1]
    h = delay_embed(ytr, lags)
    t = np.arange(h.shape[1], dtype=float)
    model = SSMLearn(t=[t], x=[h], derive_embdedding=False, ssm_dim=2,
                     dynamics_type="map", dynamics_structure="generic", error_metric="NTE")
    model.get_reduced_coordinates("linearchart")
    model.get_parametrization(poly_degree=3, alpha=1e-4, cv=0, do_scaling=True, fit_intercept=False)
    model.get_reduced_dynamics(poly_degree=dyn_degree, alpha=1e-4, cv=0, do_scaling=True,
                               fit_intercept=False)
    h0 = h[:, -1]                                  # delay vector ending at the last sample
    x_ic = np.tile(h0[:, None], (1, n_run + 1))    # predict() advects from column 0
    pred = model.predict(t=[np.arange(n_run + 1, dtype=float)], x=[x_ic])
    obs = pred["observables"][0]
    fc = obs[-d:, 1 : n_run + 1].T                 # newest block, steps 1..n_run
    bad = ~np.isfinite(fc).all(axis=1)
    if bad.any():
        fc[int(np.argmax(bad)):] = np.nan
    big = np.abs(fc) > 1e6
    if big.any():
        fc[int(np.argmax(big.any(axis=1))):] = np.nan
    return fc


def main() -> None:
    for f in sorted(DIR.glob("data_*.npz")):
        tag = f.stem.removeprefix("data_")
        out = DIR / f"ssm_{tag}.npz"
        if out.exists():
            continue
        D = np.load(f)
        ytr, yte, P, n_run = D["Ytr"], D["Yte"], float(D["P"]), int(D["n_run"])
        lags = max(4, int(round(P / 4)))
        rows = []
        fcs = []
        for deg in (5, 9):
            try:
                fc = fit_free_run(ytr, lags, n_run, deg)
            except Exception as err:            # a failed fit counts as a failure
                print(f"{tag} deg{deg}: fit failed ({err})")
                fc = np.full((n_run, ytr.shape[1]), np.nan)
            s = score(fc, yte, ytr, P)
            rows.append((deg, s))
            fcs.append(fc[: int(round(8 * P))])
            print(f"{tag:22s} SSM deg{deg} lags{lags} | e1 {s['e1']:.2f} e3 {s['e3']:.2f} "
                  f"e5 {s['e5']:.2f} | amp {s['amp']:.2f} per {s['per']:.2f} {s['fate']}", flush=True)
        np.savez(out, degrees=np.array([r[0] for r in rows]),
                 e1=np.array([r[1]["e1"] for r in rows]), e3=np.array([r[1]["e3"] for r in rows]),
                 e5=np.array([r[1]["e5"] for r in rows]), amp=np.array([r[1]["amp"] for r in rows]),
                 per=np.array([r[1]["per"] for r in rows]),
                 fate_code=np.array([FATES.index(r[1]["fate"]) for r in rows]),
                 fc=np.stack(fcs), lags=lags)


if __name__ == "__main__":
    main()

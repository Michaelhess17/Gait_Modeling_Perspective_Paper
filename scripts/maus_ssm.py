#!/usr/bin/env python
"""SSMLearnPy on the long running trials (Figure 7), with the splits, horizons and scores of maus_learning_all.jl
(deviation) and maus_ladder.jl (whole signal). Data from maus_ssm_export.jl (cache/maus/ssm_input/trial_NN.npz).

Model: Figure 5's SSMLearnPy configuration (ssm_self_driving.py): delay embedding of the training-mean-centred
channels, linear chart (leading singular vectors), cubic parametrisation of the manifold, polynomial reduced map
z_{k+1} = R(z_k), ridge 1e-4 with feature scaling, no intercept. Forecast from the delay vector ending at the start:
encode, iterate the map, decode, read the newest block. Settings chosen on an inner split of the training strides
(fit on the first half, forecast skill at 6-50 samples ahead on the second; maus_learning.jl), then refitted:
    delay window lags + 1 with lags in {6, 12, 25, 50} (Figure 5: a quarter cycle, 25), and 100 on the whole signal
    (manifold dimension, map degree) in {(2, 3), (2, 5), (2, 7), (2, 9), (4, 3), (4, 5), (6, 3), (8, 2), (12, 2), (16, 2)}
    (Figure 5: dimension 2, degree 5 or 9; the higher dimensions because the deviation is not low-dimensional: on one
    trial a 2-D manifold reconstructs 12-41% of the deviation's current sample, and maps of degree 5 or more diverge)
    ridge in {1e-4, 100} on the scaled features (Figure 5: 1e-4; the stronger ridge keeps the quadratic maps on
    8-16 dimensions from diverging over half a stride)
Representations:
    dev     deviation from the 61-stride moving-average stride (cycle and stride timing given), every training size;
            measurement noise eta: fitted and selected on the noisy deviation, started from it, scored on the clean one
    rt, sn  whole signal, real time and stride-normalised, 150 training strides
Score: skill against the average stride, 1 - sum ||yhat - y||^2 / sum ||d||^2 (d: deviation from the moving-average
stride), forecasts clipped to the training range +- half its width (maus_ladder.jl; a diverging run counts as a large,
bounded error).

    .pixi/envs/default/bin/python scripts/maus_ssm.py      -> cache/maus/ssm.npz
"""
from __future__ import annotations

import logging
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from ssmlearnpy import SSMLearn  # noqa: E402

warnings.filterwarnings("ignore")
for name in ("", "SSMLearn", "ridge_regression"):
    logging.getLogger(name).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
NPH, STEP, HISTL, HISTW = 100, 5, 50, 110
SIZES = [4, 8, 16, 32, 64, 150]
HZ = np.array([6, 12, 19, 25, 38, 50, 75, 100])
HV = [0, 1, 3, 5]                                  # 6, 12, 25, 50 samples
ETAS = [("D", 0.0), ("Dn05", 0.5), ("Dn10", 1.0)]
LAGS = [6, 12, 25, 50]
DIMDEG = [(2, 3), (2, 5), (2, 7), (2, 9), (4, 3), (4, 5), (6, 3), (8, 2), (12, 2), (16, 2)]
RIDGE = [1e-4, 1e2]
SETTINGS = [(l, m, k, a) for l in LAGS for (m, k) in DIMDEG for a in RIDGE]
SETTINGS_WHOLE = SETTINGS + [(100, m, k, a) for (m, k) in DIMDEG for a in RIDGE]   # up to a stride, as the whole-signal DMD


def delay_embed(x: np.ndarray, lags: int) -> np.ndarray:
    """Columns are delay vectors, oldest block first (ssm_self_driving.py)."""
    return np.vstack([x[i: x.shape[0] - lags + i].T for i in range(lags + 1)])


def fit(Y, rows, s):
    lags, dim, deg, a = s
    mu = Y[rows].mean(axis=0)
    h = delay_embed(Y[rows] - mu, lags)
    model = SSMLearn(t=[np.arange(h.shape[1], dtype=float)], x=[h], derive_embdedding=False, ssm_dim=dim,
                     dynamics_type="map", dynamics_structure="generic", error_metric="NTE")
    model.get_reduced_coordinates("linearchart")
    model.get_parametrization(poly_degree=3, alpha=a, cv=0, do_scaling=True, fit_intercept=False)
    model.get_reduced_dynamics(poly_degree=deg, alpha=a, cv=0, do_scaling=True, fit_intercept=False)
    return model, mu, lags


def forecast(M, Y, ts, hidx, lo, hi):
    """Forecasts (len(hidx) x starts x channels) from the delay vectors ending at ts (0-based)."""
    model, mu, lags = M
    p = Y.shape[1]
    H0 = np.stack([(Y[t - lags: t + 1] - mu).reshape(-1) for t in ts], axis=1)   # oldest block first
    z = model.encoder.predict(H0)
    out = np.zeros((len(hidx), len(ts), p))
    k = 0
    with np.errstate(all="ignore"):
        for s in range(1, HZ[hidx[-1]] + 1):
            z = model.reduced_dynamics.predict(np.nan_to_num(z.T, nan=1e6, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)).T
            if s == HZ[hidx[k]]:
                P = model.decoder.predict(z.T)[:, -p:] + mu
                P[~np.isfinite(P)] = 0.0
                out[k] = np.clip(P, lo, hi)
                k += 1
    return out


def skill(F, Y, Dv, ts, hidx):
    return np.array([1 - np.sum((F[j] - Y[ts + HZ[k]]) ** 2) / np.sum(Dv[ts + HZ[k]] ** 2) for j, k in enumerate(hidx)])


def bounds(Y, rows):
    w = Y[rows].max(axis=0) - Y[rows].min(axis=0)
    return Y[rows].min(axis=0) - 0.5 * w, Y[rows].max(axis=0) + 0.5 * w


def select_and_score(Yfit, Dfit, Yscore, Dscore, rows, inner, vst, ts, settings=SETTINGS):
    """Choose a setting on (inner, vst) of Yfit (skill against Dfit), refit on rows, forecast from Yfit histories and
    score against Yscore, Dscore."""
    lo, hi = bounds(Yfit, rows)
    vals = []
    for s in settings:
        try:
            M = fit(Yfit, inner, s)
            vals.append(np.mean(skill(forecast(M, Yfit, vst, HV, lo, hi), Yfit, Dfit, vst, HV)))
        except Exception:
            vals.append(-np.inf)
    if not np.isfinite(np.max(vals)):
        return None
    s = settings[int(np.argmax(vals))]
    F = forecast(fit(Yfit, rows, s), Yfit, ts, list(range(len(HZ))), lo, hi)
    return skill(F, Yscore, Dscore, ts, list(range(len(HZ)))), s


def trial(path: Path):
    Z = np.load(path)
    n = int(Z["nstr"])
    nh = n // 2
    out = {"dev": {}, "whole": {}}
    # deviation: stride-normalised, NPH samples per stride
    D = Z["D"]
    ts = np.arange(nh * NPH + HISTL, n * NPH - HZ[-1], STEP)
    for key, eta in ETAS:
        Dn = Z[key]
        for ntr in (SIZES if eta == 0 else [150]):
            if ntr > nh:
                continue
            rows = slice(0, ntr * NPH)
            inner = slice(0, (ntr // 2) * NPH)
            vst = np.arange((ntr // 2) * NPH + HISTL, ntr * NPH - HZ[HV[-1]], 5)
            r = select_and_score(Dn, Dn, D, D, rows, inner, vst, ts)
            if r is not None:
                out["dev"][(eta, ntr)] = r
    # whole signal, 150 training strides
    for rung in ("rt", "sn"):
        Y, Dv, b = Z["Y" + rung], Z["D" + rung], Z["bidx_" + rung] - 1
        ntr = 150
        rows = slice(0, b[ntr])
        inner = slice(0, b[ntr // 2])
        vst = np.arange(b[ntr // 2] + HISTW, b[ntr] - HZ[HV[-1]], STEP)
        ts_w = np.arange(b[nh] + HISTW, Y.shape[0] - HZ[-1], STEP)
        r = select_and_score(Y, Dv, Y, Dv, rows, inner, vst, ts_w, SETTINGS_WHOLE)
        if r is not None:
            out["whole"][rung] = r
    return path.stem, out


def main() -> None:
    files = sorted((ROOT / "cache/maus/ssm_input").glob("trial_*.npz"))
    with Pool(min(len(files), 22)) as pool:
        res = dict(pool.map(trial, files))
    names = [f.stem for f in files]
    nt = len(names)
    dev = np.full((nt, len(ETAS), len(SIZES), len(HZ)), np.nan)
    dev_set = np.full((nt, len(ETAS), len(SIZES), 4), -1.0)
    whole = np.full((nt, 2, len(HZ)), np.nan)
    whole_set = np.full((nt, 2, 4), -1.0)
    for i, nm in enumerate(names):
        for (eta, ntr), (r2, s) in res[nm]["dev"].items():
            e = [x[1] for x in ETAS].index(eta)
            dev[i, e, SIZES.index(ntr)] = r2
            dev_set[i, e, SIZES.index(ntr)] = s
        for j, rung in enumerate(("rt", "sn")):
            if rung in res[nm]["whole"]:
                whole[i, j], whole_set[i, j] = res[nm]["whole"][rung][0], res[nm]["whole"][rung][1]
    np.savez(ROOT / "cache/maus/ssm.npz", dev=dev, dev_setting=dev_set, whole=whole, whole_setting=whole_set,
             sizes=np.array(SIZES), etas=np.array([x[1] for x in ETAS]), hz=HZ, trials=np.array([int(nm.split("_")[1]) for nm in names]))
    med = lambda a: np.nanmedian(a, axis=0)
    for k, ntr in enumerate(SIZES):
        m = med(dev[:, 0, k])
        print(f"ssm dev {ntr:3d} strides: R2 1/8 / 1/4 / 1/2 stride {m[1]:.2f} / {m[3]:.2f} / {m[5]:.2f}")
    for e in (1, 2):
        m = med(dev[:, e, -1])
        print(f"ssm dev 150 strides, eta {ETAS[e][1]}: 1/8 stride {m[1]:.2f}")
    for j, rung in enumerate(("rt", "sn")):
        m = med(whole[:, j])
        print(f"ssm {rung}: R2 of the deviation 1/8 / 1/4 / 1 stride {m[1]:.2f} / {m[3]:.2f} / {m[-1]:.2f}")
    print("settings chosen (lags, dim, degree, ridge), dev 150 strides:", [tuple(x) for x in dev_set[:, 0, -1]])
    print("settings chosen, whole sn:", [tuple(x) for x in whole_set[:, 1]])


if __name__ == "__main__":
    main()

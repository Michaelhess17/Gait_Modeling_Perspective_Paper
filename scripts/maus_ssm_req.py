#!/usr/bin/env python
"""SSMLearnPy for the requirements analysis (maus_requirements.jl): the running deviation, 150 training strides, no
added noise, settings chosen on the inner split as maus_ssm.py. Per trial: forecasts at 1/8 and 1/4 stride from every
test start (the starts of maus_requirements.jl), the skill at every horizon, and free runs of 30 strides from 40
starts (unclipped; the deviation's RMS at the end of each stride over its SD on the training strides; inf if the run
leaves the finite range). SSMLearnPy has no random seed: the same data and settings give the same fit.

    .pixi/envs/default/bin/python scripts/maus_ssm_req.py   -> cache/maus/req/ssm_trial_NN.npz
"""
from __future__ import annotations

import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import maus_ssm as m

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache/maus/req"
NTR, NLONG, NSTART_LONG = 150, 30, 40


def long_runs(M, Y, tl, sd):
    model, mu, lags = M
    p = Y.shape[1]
    H0 = np.stack([(Y[t - lags: t + 1] - mu).reshape(-1) for t in tl], axis=1)
    z = model.encoder.predict(H0)
    out = np.full((len(tl), NLONG), np.inf)
    with np.errstate(all="ignore"):
        for s in range(1, NLONG * m.NPH + 1):
            z = model.reduced_dynamics.predict(np.nan_to_num(z.T, nan=1e6, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)).T
            if s % m.NPH == 0:
                P = model.decoder.predict(z.T)[:, -p:] + mu
                ok = np.isfinite(P).all(axis=1) & (np.abs(P).max(axis=1) < 1e5)
                out[ok, s // m.NPH - 1] = np.sqrt(np.mean(P[ok] ** 2, axis=1)) / sd
    return out


def trial(path: Path):
    Z = np.load(path)
    n = int(Z["nstr"]); nh = n // 2; D = Z["D"]
    ts = np.arange(nh * m.NPH + m.HISTL, n * m.NPH - m.HZ[-1], m.STEP)
    tl = ts[np.round(np.linspace(0, len(ts) - 1, NSTART_LONG)).astype(int)]
    rows = slice(0, NTR * m.NPH); inner = slice(0, (NTR // 2) * m.NPH)
    vst = np.arange((NTR // 2) * m.NPH + m.HISTL, NTR * m.NPH - m.HZ[m.HV[-1]], 5)
    lo, hi = m.bounds(D, rows)
    vals = []
    for s in m.SETTINGS:
        try:
            vals.append(np.mean(m.skill(m.forecast(m.fit(D, inner, s), D, vst, m.HV, lo, hi), D, D, vst, m.HV)))
        except Exception:
            vals.append(-np.inf)
    s = m.SETTINGS[int(np.argmax(vals))]
    M = m.fit(D, rows, s)
    F = m.forecast(M, D, ts, list(range(len(m.HZ))), lo, hi)
    r2 = m.skill(F, D, D, ts, list(range(len(m.HZ))))
    sd = np.sqrt(np.mean(D[rows] ** 2))
    np.savez(OUT / f"ssm_{path.stem}.npz", r2=r2, fc12=F[1].T, fc25=F[3].T, long=long_runs(M, D, tl, sd), setting=np.array(s, dtype=float))
    return path.stem, r2[1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "cache/maus/ssm_input").glob("trial_*.npz"))
    with Pool(min(len(files), 16)) as pool:
        res = pool.map(trial, files)
    print("ssm: remaining variance 1/8 stride, median %.2f" % np.median([1 - r for _, r in res]))


if __name__ == "__main__":
    main()

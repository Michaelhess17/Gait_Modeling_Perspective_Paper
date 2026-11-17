#!/usr/bin/env python
"""SSMLearnPy on the structure-recovery test of ltp_recovery.jl (R5). Each simulated record (cache/recovery/ssm_io):
settings from maus_ssm.py's grid chosen on an inner split (fit on the first half, forecast skill 6-50 samples ahead on
the second), refitted on all of it. SSMLearnPy's reduced map is autonomous, so its multipliers per cycle are the
eigenvalues of the linear part of the map raised to the 100th power (fewer than three if the manifold has dimension
2; missing ones count as 0). Error: as ltp_recovery.jl (three largest, best matching).

    .pixi/envs/default/bin/python scripts/ltp_recovery_ssm.py   -> cache/recovery/ssm.npz
"""
from __future__ import annotations

import itertools
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import maus_ssm as m

warnings.filterwarnings("ignore")
DIR = Path(__file__).resolve().parent.parent / "cache/recovery"


def mult_err(e, rho):
    e = np.array(sorted(e, key=abs, reverse=True)[:3] + [0.0] * max(0, 3 - len(e)), dtype=complex)
    return min(np.mean(np.abs(e[list(p)] - rho)) for p in itertools.permutations(range(3)))


def one(path: Path):
    Z = np.load(path); D = Z["Z"]; rho = Z["rho"]
    n = D.shape[0]; inner = slice(0, n // 2)
    vst = np.arange(n // 2 + m.HISTL, n - m.HZ[m.HV[-1]], 5)
    lo, hi = m.bounds(D, slice(0, n))
    best, sb = -np.inf, None
    for s in m.SETTINGS:
        try:
            v = np.mean(m.skill(m.forecast(m.fit(D, inner, s), D, vst, m.HV, lo, hi), D, D, vst, m.HV))
        except Exception:
            continue
        if np.isfinite(v) and v > best:
            best, sb = v, s
    model, mu, lags = m.fit(D, slice(0, n), sb)
    dim = sb[1]
    L = model.reduced_dynamics.map_info["coefficients"][:, :dim]
    e = np.linalg.eigvals(L) ** m.NPH
    nc, rep = (int(x) for x in path.stem.split("_")[1:3])
    return nc, rep, e, mult_err(list(e), rho), sb


def main() -> None:
    files = sorted((DIR / "ssm_io").glob("sim_*.npz"))
    with Pool(min(len(files), 16)) as pool:
        res = pool.map(one, files)
    nc = np.array([r[0] for r in res]); err = np.array([r[3] for r in res])
    for c in sorted(set(nc)):
        print(f"ssm {c:3d} cycles: multiplier error {np.median(err[nc == c]):.3f}")
    mult = np.zeros((len(res), 3), dtype=complex)
    for i, r in enumerate(res):
        e = sorted(r[2], key=abs, reverse=True)[:3]
        mult[i, :len(e)] = e
    np.savez(DIR / "ssm.npz", nc=nc, rep=np.array([r[1] for r in res]), err=err, mult=mult)


if __name__ == "__main__":
    main()

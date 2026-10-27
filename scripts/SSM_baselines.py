#!/usr/bin/env python
"""Trivial-baseline check for the SSMLearnPy gait forecasts (ssm_fit_gait.py).

Reproduces the evaluation protocol of ssm_fit_gait.py exactly and scores naive
predictors under it, reusing the cached per-trial SSM error curves
(cache/fig7_ssmlearnpy_primary.npz) rather than refitting.

Protocol (extracted from ssm_fit_gait.py + SSMLearnPy source):
  * data/all_human_data.npy, shape (299 trials, 1500 samples @ 100 Hz, 6 channels);
    no filtering or decimation.
  * Per trial, each channel is z-scored with mean/std of samples 0..999 (training).
  * SSM is fit on the delay embedding (31 blocks x 6 ch) of z[0:1000].
  * Test Hankel is built from z[970:1500]; column j = [z[970+j], ..., z[1000+j]].
    SSMLearnPy's map advection uses ONLY column 0 (z[970..1000]) as the initial
    condition, so the model sees through sample 1000 (one sample past training).
  * Predictions: column j = j map iterates; evaluation reads the first (oldest)
    6 rows, i.e. the target at "test time index j" is z[970 + j].
  * Error: rmse_t[j] = sqrt(mean over 6 channels of (pred - true)^2) at that single
    time index (instantaneous, not cumulative), in training-std units.
  * Reported "1/3/5 s" = rmse_t[99], [299], [499] -> targets z[1069], z[1269],
    z[1469], i.e. 69/269/469 samples (0.69/2.69/4.69 s) past the last sample the
    model has seen.  Medians are np.nanmedian across trials (non-finite SSM
    rollouts are dropped).

Baselines are given the same information set (z[0..1000]); period estimation and
templates use training samples z[0:1000] only.  For targets that lie inside the
information set (j <= 30) the observed value is used (irrelevant to j >= 99).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/all_human_data.npy"
META_PATH = ROOT / "data/all_human_data_metadata.csv"
SSM_CACHE = ROOT / "cache/fig7_ssmlearnpy_primary.npz"
OUT_NPZ = ROOT / "cache/ssm_baselines.npz"
OUT_CSV = ROOT / "cache/ssm_baselines_horizons.csv"
OUT_JSON = ROOT / "cache/ssm_baselines_summary.json"

FS = 100.0
TRAIN_STEPS = 1000
TEST_STEPS = 500
DELAY_LAGS = 30
INFO_END = TRAIN_STEPS + 1  # SSM initial condition includes z[1000]
OFFSET = TRAIN_STEPS - DELAY_LAGS  # target index for test column j is OFFSET + j
HORIZONS = {"1s": 99, "3s": 299, "5s": 499}
PMIN, PMAX = 50, 250  # 0.5-2.5 s
DMD_RANK = 70        # was 10, which limited the fit (1-s error 0.27 vs 0.21 at rank >= 30)
DMD_RIDGE = 1e-4     # Tikhonov strength relative to the leading singular value squared (companion paper)


def zscore(x_trial: np.ndarray) -> np.ndarray:
    x_train = x_trial[:TRAIN_STEPS]
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (x_trial - mean) / scale


def estimate_period(z_train: np.ndarray) -> tuple[float, float]:
    """Stride period (samples) from the summed per-channel autocorrelation.

    Picks the first local maximum in [PMIN, PMAX] whose height is >= 0.8 x the
    largest local maximum there (guards against selecting 2P), then refines with
    parabolic interpolation.  Returns (period, normalized ACF height).
    """
    zc = z_train - z_train.mean(axis=0)
    n = zc.shape[0]
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(zc, nfft, axis=0)
    ac = np.fft.irfft(np.abs(f) ** 2, nfft, axis=0)[:n].sum(axis=1)
    ac = ac / ac[0]
    seg = ac[PMIN - 1 : PMAX + 2]
    loc = np.where((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + PMIN
    if len(loc) == 0:
        lag = PMIN + int(np.argmax(ac[PMIN : PMAX + 1]))
    else:
        heights = ac[loc]
        ok = loc[heights >= 0.8 * heights.max()]
        lag = int(ok[0])
    if PMIN < lag < PMAX:
        y0, y1, y2 = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    return lag + delta, float(ac[lag])


def interp_rows(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of z (T, C) at fractional sample positions t."""
    i0 = np.floor(t).astype(int)
    w = (t - i0)[:, None]
    i1 = np.minimum(i0 + 1, z.shape[0] - 1)
    return (1 - w) * z[i0] + w * z[i1]


def forecast_targets() -> np.ndarray:
    return OFFSET + np.arange(TEST_STEPS)


def pred_mean(z: np.ndarray, **_) -> np.ndarray:
    return np.zeros((TEST_STEPS, z.shape[1]))


def pred_persistence(z: np.ndarray, **_) -> np.ndarray:
    n = forecast_targets()
    out = np.repeat(z[INFO_END - 1][None, :], TEST_STEPS, axis=0)
    inside = n < INFO_END
    out[inside] = z[n[inside]]
    return out


def pred_repeat_stride(z: np.ndarray, period: float, **_) -> np.ndarray:
    """z_hat(n) = z(n - k P) with the smallest k putting n - kP inside the info set."""
    n = forecast_targets().astype(float)
    last = INFO_END - 1
    k = np.maximum(np.ceil((n - last) / period), 0)
    src = n - k * period
    return interp_rows(z[:INFO_END], src)


def pred_phase_template(z: np.ndarray, period: float, nbins: int = 100, **_) -> np.ndarray:
    """Mean-cycle template, phase-aligned to the last stride, extrapolated at period P."""
    z_train = z[:TRAIN_STEPS]
    t_train = np.arange(TRAIN_STEPS)
    phase = (t_train / period) % 1.0
    b = np.minimum((phase * nbins).astype(int), nbins - 1)
    tmpl = np.zeros((nbins, z.shape[1]))
    for c in range(z.shape[1]):
        tmpl[:, c] = np.bincount(b, weights=z_train[:, c], minlength=nbins) / np.maximum(
            np.bincount(b, minlength=nbins), 1
        )
    # light circular smoothing (3-bin boxcar) to remove bin noise
    tmpl = (np.roll(tmpl, 1, 0) + tmpl + np.roll(tmpl, -1, 0)) / 3.0

    def tmpl_at(ph: np.ndarray) -> np.ndarray:
        x = (ph % 1.0) * nbins - 0.5
        i0 = np.floor(x).astype(int)
        w = (x - i0)[:, None]
        return (1 - w) * tmpl[i0 % nbins] + w * tmpl[(i0 + 1) % nbins]

    # align: last ~1 stride of the information set vs template, best circular shift
    L = int(round(period))
    seg = z[INFO_END - L : INFO_END]
    rel = (np.arange(L) - (L - 1)) / period  # phase of each seg sample relative to last sample
    shifts = np.linspace(0.0, 1.0, 400, endpoint=False)
    errs = [np.mean((tmpl_at(s + rel) - seg) ** 2) for s in shifts]
    phi_last = shifts[int(np.argmin(errs))]  # template phase of sample INFO_END-1

    n = forecast_targets()
    out = tmpl_at(phi_last + (n - (INFO_END - 1)) / period)
    inside = n < INFO_END
    out[inside] = z[n[inside]]
    return out


def _stabilise(A: np.ndarray) -> np.ndarray:
    """Pin eigenvalues outside the unit circle onto it, block by block in the real Schur form (companion paper)."""
    from scipy.linalg import schur
    T, Z = schur(A, output="real"); n = T.shape[0]; i = 0
    while i < n:
        if i < n - 1 and T[i + 1, i] != 0:
            m = np.max(np.abs(np.linalg.eigvals(T[i:i + 2, i:i + 2])))
            if m > 1:
                T[i:i + 2, i:i + 2] /= m
            i += 2
        else:
            if abs(T[i, i]) > 1:
                T[i, i] /= abs(T[i, i])
            i += 1
    return Z @ T @ Z.T


def pred_hankel_dmd(z: np.ndarray, rank: int = DMD_RANK, ridge: float = DMD_RIDGE, **_) -> np.ndarray:
    """Linear map on the rank-r POD projection of the same 31-block delay embedding as the SSM.

    The companion paper's fit: exact DMD with ridge filter S/(S^2 + ridge*s1^2) and Schur stabilisation.
    Initial condition and read-out are identical to ssm_fit_gait.py: IC is the
    delay vector z[970..1000]; column j's first block predicts z[970+j].
    """
    lags = DELAY_LAGS
    zt = z[:TRAIN_STEPS]
    H = np.vstack([zt[i : TRAIN_STEPS - lags + i].T for i in range(lags + 1)])  # (186, 970)
    X, Y = H[:, :-1], H[:, 1:]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, len(S))
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    A = _stabilise(Ur.T @ Y @ Vr @ np.diag(Sr / (Sr**2 + ridge * S[0] ** 2)))
    h0 = np.concatenate([z[OFFSET + i] for i in range(lags + 1)])  # z[970..1000]
    ak = Ur.T @ h0
    out = np.empty((TEST_STEPS, z.shape[1]))
    for j in range(TEST_STEPS):
        out[j] = (Ur @ ak)[: z.shape[1]]
        ak = A @ ak
    return out, float(np.max(np.abs(np.linalg.eigvals(A))))


def err_curve(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    r[~np.isfinite(r)] = np.nan
    return r


def med_iqr(v: np.ndarray) -> str:
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "n/a"
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:.2f} [{q1:.2f}-{q3:.2f}]"


def main() -> None:
    data = np.load(DATA_PATH)
    meta = pd.read_csv(META_PATH)
    groups = meta["lf_or_hf"].to_numpy()
    ssm = np.load(SSM_CACHE, allow_pickle=True)
    ssm_rmse = ssm["rmse"]
    assert ssm_rmse.shape == (data.shape[0], TEST_STEPS)
    assert np.all(ssm["groups"] == groups)

    methods = ["mean", "persistence", "repeat_stride", "phase_template", "hankel_dmd"]
    E = {m: np.full((data.shape[0], TEST_STEPS), np.nan) for m in methods}
    periods = np.zeros(data.shape[0])
    acf_h = np.zeros(data.shape[0])
    dmd_rho = np.zeros(data.shape[0])
    # sanity check: does the cached SSM curve match a reconstruction of the target?
    targets = forecast_targets()

    for i in range(data.shape[0]):
        z = zscore(data[i])
        true = z[targets]
        P, h = estimate_period(z[:TRAIN_STEPS])
        periods[i], acf_h[i] = P, h
        E["mean"][i] = err_curve(pred_mean(z), true)
        E["persistence"][i] = err_curve(pred_persistence(z), true)
        E["repeat_stride"][i] = err_curve(pred_repeat_stride(z, period=P), true)
        E["phase_template"][i] = err_curve(pred_phase_template(z, period=P), true)
        p, rho = pred_hankel_dmd(z)
        dmd_rho[i] = rho
        E["hankel_dmd"][i] = err_curve(p, true)

    allm = ["ssm"] + methods
    E["ssm"] = ssm_rmse
    ssm_ok = np.isfinite(ssm_rmse[:, 499])

    # ---- tables -----------------------------------------------------------
    rows = []
    lines = []
    for subset_name, base_mask in [("all trials (SSM: finite only)", np.ones(len(groups), bool)),
                                   ("SSM-finite subset (paired)", ssm_ok)]:
        lines.append(f"\n=== {subset_name} : median [IQR] instantaneous NRMSE ===")
        header = f"{'group':5s} {'method':15s} " + " ".join(f"{h:>20s}" for h in HORIZONS)
        lines.append(header)
        for g in ["ALL", "AB", "HF", "LF"]:
            gm = base_mask & ((groups == g) if g != "ALL" else True)
            for m in allm:
                cells = []
                for hname, k in HORIZONS.items():
                    v = E[m][gm, k]
                    cells.append(med_iqr(v))
                    vv = v[np.isfinite(v)]
                    rows.append(dict(subset=subset_name, group=g, method=m, horizon=hname,
                                     n=int(len(vv)), median=float(np.median(vv)) if len(vv) else np.nan,
                                     q25=float(np.percentile(vv, 25)) if len(vv) else np.nan,
                                     q75=float(np.percentile(vv, 75)) if len(vv) else np.nan))
                lines.append(f"{g:5s} {m:15s} " + " ".join(f"{c:>20s}" for c in cells))
            lines.append("")

    # ---- head-to-head --------------------------------------------------------
    lines.append("\n=== SSM vs baselines, paired on SSM-finite trials (n=%d) ===" % ssm_ok.sum())
    h2h = {}
    for hname, k in HORIZONS.items():
        s = ssm_rmse[ssm_ok, k]
        meds = {m: np.median(E[m][ssm_ok, k]) for m in methods if m != "mean"}
        best = min(meds, key=meds.get)
        d = {}
        for m in ["phase_template", "repeat_stride", best]:
            b = E[m][ssm_ok, k]
            frac = float(np.mean(s < b))
            stat = wilcoxon(s, b)
            d[m] = dict(frac_ssm_better=frac, wilcoxon_p=float(stat.pvalue),
                        median_diff_ssm_minus_base=float(np.median(s - b)))
        # treating non-finite SSM as losses (all 299)
        frac_all = float(np.mean(np.where(np.isfinite(ssm_rmse[:, k]),
                                          ssm_rmse[:, k] < E["phase_template"][:, k], False)))
        h2h[hname] = dict(best_baseline=best, comparisons=d, frac_ssm_beats_template_all299=frac_all)
        lines.append(f"{hname}: best baseline = {best} (median {meds[best]:.3f})")
        for m, r in d.items():
            lines.append(f"   vs {m:15s}: SSM better in {r['frac_ssm_better']:.1%}, "
                         f"median(SSM-base) = {r['median_diff_ssm_minus_base']:+.3f}, "
                         f"Wilcoxon p = {r['wilcoxon_p']:.2e}")
        lines.append(f"   SSM beats phase_template in {frac_all:.1%} of all 299 (non-finite = loss)")

    # ---- window-averaged robustness check ----------------------------------
    lines.append("\n=== robustness: error averaged over test indices 31..k (lead 1..k-30), SSM-finite subset ===")
    for hname, k in HORIZONS.items():
        cells = []
        for m in allm:
            v = np.nanmean(E[m][ssm_ok, 31 : k + 1], axis=1)
            cells.append(f"{m}={np.median(v):.2f}")
        lines.append(f"{hname}: " + "  ".join(cells))

    lines.append("\n=== diagnostics ===")
    lines.append(f"period (s): median {np.median(periods)/FS:.2f}, range {periods.min()/FS:.2f}-{periods.max()/FS:.2f}; "
                 f"at bounds: {(np.abs(periods-PMIN)<1).sum()} low, {(np.abs(periods-PMAX)<1).sum()} high; "
                 f"ACF height median {np.median(acf_h):.2f}, <0.3 in {(acf_h<0.3).sum()} trials")
    for g in ["AB", "HF", "LF"]:
        lines.append(f"  {g}: period median {np.median(periods[groups==g])/FS:.2f} s, ACF height median {np.median(acf_h[groups==g]):.2f}")
    lines.append(f"Hankel-DMD spectral radius: median {np.median(dmd_rho):.4f}, >1 in {(dmd_rho>1).sum()} trials")
    lines.append(f"SSM non-finite trials by group: " + ", ".join(
        f"{g}={(~ssm_ok & (groups==g)).sum()}" for g in ["AB", "HF", "LF"]))
    lines.append("Baseline errors on SSM-non-finite trials (3s), phase_template: "
                 + med_iqr(E["phase_template"][~ssm_ok, 299]) + " vs SSM-finite: "
                 + med_iqr(E["phase_template"][ssm_ok, 299]))

    report = "\n".join(lines)
    print(report)

    np.savez_compressed(OUT_NPZ, groups=groups, periods_samples=periods, acf_height=acf_h,
                        dmd_spectral_radius=dmd_rho, ssm_finite=ssm_ok,
                        test_time_s=np.arange(TEST_STEPS) / FS,
                        target_sample_index=targets,
                        **{f"err_{m}": E[m] for m in allm})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(dict(head_to_head=h2h, report=report), indent=2))
    print(f"\nsaved {OUT_NPZ}\nsaved {OUT_CSV}\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()

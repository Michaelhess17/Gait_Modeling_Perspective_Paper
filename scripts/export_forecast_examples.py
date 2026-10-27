#!/usr/bin/env python
"""Export example forecasts (all methods) for the forecasting-baseline figure.

Reuses the exact protocol in SSM_baselines.py (same z-scoring, period estimate,
initial information and target indexing) for the three representative trials
already stored in the SSM cache, and writes plain numeric arrays that NPZ.jl can
read (no object dtypes).

    pixi run python export_forecast_examples.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import SSM_baselines as B

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache/fig_forecast_examples.npz"


def main() -> None:
    data = np.load(B.DATA_PATH)
    meta = pd.read_csv(B.META_PATH)
    ssm = np.load(B.SSM_CACHE, allow_pickle=True)
    idx = ssm["example_indices"]
    targets = B.forecast_targets()

    out = {"example_indices": idx.astype(np.int64), "targets": targets.astype(np.int64)}
    group_code = {"AB": 0, "HF": 1, "LF": 2}
    out["group_code"] = np.array([group_code[g] for g in meta["lf_or_hf"]], dtype=np.int64)
    out["speed"] = meta["speed"].to_numpy(dtype=float)

    for k, (g, i) in enumerate(zip(["AB", "HF", "LF"], idx)):
        z = B.zscore(data[i].astype(float))
        period, _ = B.estimate_period(z[: B.TRAIN_STEPS])
        out[f"ex{k}_z"] = z                                  # full z-scored trial
        out[f"ex{k}_true"] = z[targets]                      # scored targets
        out[f"ex{k}_template"] = B.pred_phase_template(z, period=period)
        out[f"ex{k}_repeat"] = B.pred_repeat_stride(z, period=period)
        out[f"ex{k}_dmd"] = B.pred_hankel_dmd(z)[0]
        out[f"ex{k}_ssm"] = ssm[f"example_{g}_pred"].T       # (500, 6)
        out[f"ex{k}_period"] = np.array(period)
        # sanity: the SSM cache stores the same targets we rebuild here
        assert np.allclose(ssm[f"example_{g}_true"].T, z[targets])

    np.savez(OUT, **out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

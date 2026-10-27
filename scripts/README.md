# scripts/

Run everything from the repository root, e.g. `julia --project=. scripts/Fig_self_driving.jl`.
Paths are resolved relative to the repository root (`cache/`, `data/`, `figures/`, `logs/`).
Section and figure numbers refer to ../OUTLINE.md.

## Shared
- `common.jl`: data loading, `cache_path`, Hankel/DMD helpers (included by most scripts)
- `paper_style.jl`: CairoMakie theme, palette, `save_paper` (writes figures/*.pdf|svg|png)

## Self-driving benchmark (§3, Fig 2)
- `self_driving_benchmark.jl`: systems, regimes, neural ODE / ESN / DMD fits, free-run fates
- `dmd_tuning.jl`, `esn_tuning.jl`: train-only hyperparameter selection for DMD and ESN
- `ssm_self_driving.py`: SSMLearnPy fits on the same data (pixi environment)
- `Fig_self_driving.jl`: figure

## Forecast baselines (§3 ¶5, supplement)
- `ssm_fit_gait.py`: SSMLearnPy fits to gait (writes cache/fig7_ssmlearnpy_primary.npz)
- `SSM_baselines.py`: SSM vs mean-cycle template / repeat stride / Hankel-DMD
- `export_forecast_examples.py`: example traces for the figure
- `Fig_forecast_baselines.jl`: figure

## Learnability (§2, Fig 1)
- `gait_overparam.jl` (+ `floquet_from_field.jl`): equal-loss neural-ODE fits to gait
- `Fig_node_gait.jl`: figure panel
- `dmd_delay_sweep.jl`: DMD off-cycle relaxation vs delay window, rank and number of decays
- `Fig_dmd_datasize.jl`: figure panel (relaxation rate vs transient data)
- `offcycle_relaxation.jl` → `offcycle_v2.jl`: off-cycle relaxation test, phase-independent scoring
- `Fig_offcycle.jl`: figure panel (10-decay comparison; to be redone at matched larger data)

## Floquet multipliers from models (§4 ¶3)
Include chain: `floquet_maps_benchmark.jl` (coupled FHN system and return-map estimators)
→ `floquet_from_models.jl` (chain-of-Jacobians readout, truth validation)
→ `floquet_tuned.jl` (wide tuning, single trajectory) → `floquet_decays.jl` (decay-trained).
- `run_floquet_tuned.sh`, `run_floquet_decays.sh`: drivers for the full runs
- `Fig_floquet_models.jl`: figure; **still reads the early untuned runs and needs rebuilding from
  tuned_*/decays_* results**

## Noise and estimator bias (§6 ¶4)
- `dmd_noise_variants.jl`: least squares vs forward–backward vs TLS DMD under measurement noise
- `noise_models.jl`: DMD / ESN / neural ODE relaxation under measurement noise, clean and noisy warm-up

## Coupling direction (§2 ¶2, supplement)
- `coupling_benchmark.jl`, `summarize_coupling.jl`

## Ground truth for the new Floquet / non-normality figures (§4, Figs 3–4; in progress)
- `systems_truth.jl`: Van der Pol, Stuart–Landau and SLIP (via ~/Documents/Julia/SLIP.jl) multipliers
  and non-normality in observed coordinates

# scripts/

Run everything from the repository root, e.g. `julia --project=. scripts/Fig_self_driving.jl`.
Paths are resolved relative to the repository root (`cache/`, `data/`, `figures/`, `logs/`).
Section and figure numbers refer to ../OUTLINE.md.

## Shared
- `common.jl`: data loading, `cache_path`, Hankel/DMD helpers (included by most scripts)
- `paper_style.jl`: CairoMakie theme, palette, `save_paper` (writes figures/*.pdf|svg|png)

## Main figures (numbered as in OUTLINE.md)
- `Fig1_learnability.jl` (+ `fig1_coverage_data.jl`): data coverage and the cycle-to-cycle map, equal-loss
  gait neural ODEs, DMD relaxation vs transient data. Panel D (matched-data comparison) still to come.
- `Fig2_self_driving.jl`: free-running benchmark
- `Fig3_floquet_maps.jl` (+ `fig3_slip_maps.jl`, data from `slip_truth.jl`): data-driven Floquet maps of SLIP,
  including the instrumental-variable section map
- `Fig4_nonnormality.jl` (data from `systems_truth.jl`, `slip_truth.jl`): transient growth beyond eigenvalues
- `Fig5_gait_variability.jl` (+ `fig5_gait_data.jl`; human data from ../Gait_NonNormality_Paper3, path in
  `GAIT_P3`): mean cycle vs deviations, mean-cycle-only surrogate of the companion paper's transient growth,
  identification by feature family
- `Fig6_linear_affordances.jl` (= Fig 5; `fig6_relaxation_data.jl` → cache/fig6/relaxation.jls): per-start, per-seed
  off-cycle relaxation on coupled FHN and SLIP, scored to the curve through the reference's samples and from each
  model's settled attractor (DMD, ESN refitted; neural ODE from cached offv2 runs); multipliers of each model's own SLIP
  cycle (DMD operator in the cycle frame, ESN monodromy)
- Supplement: `FigS5_identification.jl` (from the Fig 5 cache), `FigS1_forecast_baselines.jl`, `FigS2_node_gait.jl`, `FigS3_floquet_models.jl` (needs rebuilding
  from tuned/decay results), `FigS4_offcycle.jl`
- `dmd_window_return.jl` (`-t 16`): can DMD predict the return to the true cycle on coupled FHN with a longer delay
  window or more decays (Fig 5A follow-up)
- `FigS7_validation.jl` (`validation_criteria.jl`, `-t 16`, → cache/selfdrive/validation_criteria.jls): one-step
  against rollout validation errors on the DMD and ESN tuning grids, and what each criterion would choose
- `FigS6_lattice.jl` (`figS6_slip_data.jl` → cache/figS6/slip.jls; `gait_lattice.jl` → cache/fig5/gait_lattice.jls;
  `gait_lattice_null.jl` → cache/fig5/gait_lattice_null.jls): how the lattice-constrained model works, and how it
  compares on SLIP and gait

## Self-driving benchmark (§3, Fig 2)
- `self_driving_benchmark.jl`: systems, regimes, neural ODE / ESN / DMD fits, free-run fates
- `dmd_tuning.jl`, `esn_tuning.jl`: train-only hyperparameter selection for DMD and ESN
- `ssm_self_driving.py`: SSMLearnPy fits on the same data (pixi environment)
- `Fig2_self_driving.jl`: figure

## Forecast baselines (§3 ¶5, supplement)
- `ssm_fit_gait.py`: SSMLearnPy fits to gait (writes cache/fig7_ssmlearnpy_primary.npz)
- `SSM_baselines.py`: SSM vs mean-cycle template / repeat stride / Hankel-DMD
- `export_forecast_examples.py`: example traces for the figure
- `FigS1_forecast_baselines.jl`: figure

## Learnability (§2, Fig 1)
- `gait_overparam.jl` (+ `floquet_from_field.jl`): equal-loss neural-ODE fits to gait
- `FigS2_node_gait.jl`: full gait neural-ODE figure (supplement; panels E–F reused in Fig 1B)
- `dmd_delay_sweep.jl`: DMD off-cycle relaxation vs delay window, rank and number of decays
- `offcycle_relaxation.jl` → `offcycle_v2.jl`: off-cycle relaxation test, phase-independent scoring
- `FigS4_offcycle.jl`: 10-decay off-cycle comparison (supplement; to be redone at matched larger data)

## Floquet multipliers from models (§4 ¶3)
Include chain: `floquet_maps_benchmark.jl` (coupled FHN system and return-map estimators)
→ `floquet_from_models.jl` (chain-of-Jacobians readout, truth validation)
→ `floquet_tuned.jl` (wide tuning, single trajectory) → `floquet_decays.jl` (decay-trained).
- `run_floquet_tuned.sh`, `run_floquet_decays.sh`: drivers for the full runs
- `FigS3_floquet_models.jl`: figure; **still reads the early untuned runs and needs rebuilding from
  tuned_*/decays_* results**

## Noise and estimator bias (§6 ¶4)
- `dmd_noise_variants.jl`: least squares vs forward–backward vs TLS DMD under measurement noise
- `noise_models.jl`: DMD / ESN / neural ODE relaxation under measurement noise, clean and noisy warm-up
- `fig6_bias_correction.jl`: LS / FB / TLS / IV on the SLIP section map (text only; formerly Fig 6B–D)
- `iv_gait_stride.jl`: LS vs instrumental-variable stride maps on human gait (not identifiable from 15-s trials)
- `iv_gait_operator.jl`: LS vs IV (lagged delay window as instrument) for the companion paper's operator

## Coupling direction (§2 ¶2, supplement)
- `coupling_benchmark.jl`, `summarize_coupling.jl`

## Ground truth for the Floquet / non-normality figures (§4, Figs 3–4)
- `systems_truth.jl`: Van der Pol (μ sweep), Stuart–Landau, coupled FHN: multipliers and worst-case growth
  in observed coordinates, from forward-integrated variational equations
- `slip_decays.jl` (SLIP.jl environment) → `run_slip_offcycle.sh`: SLIP (AOA 73°, k̃ 21, ρ 0.77) decay data and
  the off-cycle model comparison for Fig 6A; SLIP is a third `sys` in floquet_decays.jl / offcycle_v2.jl
- `slip_rotation_truth.jl` (SLIP.jl environment) → `slip_rotation.jl`: 30-step true off-cycle runs, and the
  section-map multiplier of truth and every model (Fig 6C)
- `slip_long_decays.jl` (SLIP.jl environment) → `floquet_lifted.jl <C|D|A|B|E>`: representing the limit cycle's
  Floquet structure in linear models (long decays, ResDMD residuals, phase-indexed monodromy, lattice-constrained
  fit, neural phase–isostable model); exploratory, §6 ¶6–7
- `slip_kicked_walks.jl` (SLIP.jl environment) → `gait_lattice.jl` (`-t 12`): lattice-constrained Floquet model on
  noise-driven, stride-normalised data, μ chosen on the person's other trials; kicked SLIP walks (validation) and
  the 299 gait trials (multiplier, forecast skill against the mean stride, a deviation DMD and the companion DMD,
  stride-matched check, mode shape); `gait_lattice_null.jl` (`-t 12`, after it): stride-shuffle null for the skill
  of μ chosen on the scored strides, which needs no selection across trials at other speeds
- `gait_overparam_validation.jl`: training loss against held-out rollout error for the gait neural ODEs (reads the
  gait_overparam caches); which fit each criterion would pick
- `slip_truth.jl`: SLIP step Jacobian and kicked walks; run in SLIP.jl's own environment
  (`julia --project=$HOME/Documents/Julia/SLIP.jl scripts/slip_truth.jl [<kappa> <seed> <nsteps>]`)

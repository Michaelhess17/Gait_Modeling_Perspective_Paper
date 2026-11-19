# scripts/

Run everything from the repository root with the project environment: `julia --project=. scripts/<path>.jl`.
Paths are resolved relative to the repository root (`cache/`, `data/`, `figures/`, `logs/`).

- **Figures** (`scripts/figures/Figure_NN.jl`) only plot: they read cached results and write
  `figures/Figure_NN.{pdf,svg,png}`. Rebuild all of them with `julia --project=. scripts/make_figures.jl`
  (or some: `scripts/make_figures.jl 05 S03`). Figures 3–9 also save their headline numbers
  (`scorecard!` in `paper_style.jl` → `cache/scorecard/`); `scorecard.jl` turns them into the comparison table
  `_scorecard.md`, included in `index.qmd` (run after a full rebuild by `make_figures.jl`, or by hand after a partial one).
- **Analyses** (the other scripts here) simulate, fit models and write the caches. Run them before the figures that
  read them; long runs note their thread count and time in their header.
- SLIP ground truth needs the SLIP.jl environment (`julia --project=$HOME/Documents/Julia/SLIP.jl …`) where noted.
- GPU scripts (LSTM): `NIX_LD_LIBRARY_PATH=/run/opengl-driver/lib julia --project=. …` on this machine.
- Older and exploratory analyses are in `archive/` (git-ignored; also in the git history).

## Figures in manuscript order

| Figure | Script | Shows | Reads | Made by |
|---|---|---|---|---|
| 1 | `figures/Figure_01.jl` | overview (schematic): walking is a cycle plus responses; the average stride and stability numbers miss the dynamics; what a model must provide (R1–R6) | — (schematic; exact toy systems) | — |
| 2 | `figures/Figure_02.jl` | walking is disturbed again before it recovers | `cache/fig1/transient.jls`, `cache/truth` | `fig1_transient_data.jl`; `slip_truth.jl` (SLIP.jl env), `systems_truth.jl` |
| 3 | `figures/Figure_03.jl` | R1: forecasting the predictable short-time response of kicked SLIP, all model classes, against its floor | `cache/predictable` | `slip_predictable_truth.jl` (SLIP.jl env) → `predictable_response.jl`, `predictable_more.jl <fip\|dmdesn\|node\|lstm\|export\|ssm>` (`lstm 1,2` fits a subset of the 8 configurations per process; `lstmmerge` combines them) (the 30-stride free runs of Figure 5 B come from the same runs), `predictable_ssm.py` (after `export`) |
| 4 | `figures/Figure_04.jl` | R1: long running recordings, whole signal against deviation over the forecast horizon | `cache/maus/learning_all_*.jls`, `cache/maus/ladder*.jls`, `cache/maus/node_*.jls`, `cache/maus/ssm.npz` (shared loader `figures/running_common.jl`) | `maus_extract.jl` (needs `data/maus_data`) → `maus_learning_all.jl cpu <η>`, `maus_learning_all.jl lstm <η>` (GPU), `maus_ladder.jl`, `maus_ladder_reference.jl`, `maus_node.jl dev <η>` / `whole` (GPU), `maus_ssm_export.jl` → `maus_ssm.py` (pixi env: `.pixi/envs/default/bin/python`) |
| 5 | `figures/Figure_05.jl` | R2: free runs — example free runs of every model class; the outcome after 30 cycles in the benchmark, on the SLIP walker and on the running whole signal; first-cycle error; ε-tube; running deviation free runs | `cache/selfdrive`, `cache/predictable/more_*.jls`, `cache/maus/req`, `cache/maus/whole_r2.jls` | `self_driving_benchmark.jl`, `dmd_tuning.jl`, `esn_tuning.jl`, `benchmark_more.jl fip\|lstm`, `ssm_self_driving.py`, `stabilization_benchmark.jl`, `r2_tube.jl` (after the others); `predictable_more.jl` and `predictable_ssm.py` (SLIP, as Figure 3); `maus_requirements.jl`, `maus_ssm_req.py`; `maus_ssm_whole_r2.py` → `maus_whole_r2.jl` |
| 6 | `figures/Figure_06.jl` | R3: reproducibility — forecasts and quantities across random seeds (running, benchmark) | `cache/maus/req`, `cache/selfdrive` | `maus_requirements.jl cpu\|node\|lstm`, `maus_ssm_req.py` |
| 7 | `figures/Figure_07.jl` | R4: realistic recordings — what a 15-s gait trial contains; training strides and noise (running); the step-map multiplier against record length and noise (SLIP); partial observation; cost per fit | `cache/fig5/gait.jls`, `cache/fig3`, and as Figures 3 and 4 | `fig5_gait_data.jl`, `fig3_slip_maps.jl`, `slip_truth.jl`, and as Figures 3 and 4 |
| 8 | `figures/Figure_08.jl` | R5: recovering known phase-dependent structure and Floquet multipliers from data, all model classes; transient growth of a whole-signal operator on gait against the mean cycle alone | `cache/recovery`, `cache/fig5/gait.jls` | `ltp_recovery.jl linear\|esn\|node\|lstm\|export`, `ltp_recovery_ssm.py`, `fig5_gait_data.jl` |
| 9 | `figures/Figure_09.jl` | R6: comparing people — a per-person quantity's intraclass correlation; when one model can serve many recordings | `cache/maus/req`, `cache/heterogeneity`, `cache/gait_group/dmd.jls`, `cache/cohort` | `maus_requirements.jl`, `gait_group_fits.jl dmd`, `heterogeneity_pools.jl`, `cohort_fits.jl <slip\|cfhn> dmd` ← `slip_family_data.jl` (SLIP.jl env), `cfhn_family_data.jl` |

## Supplementary figures

| Figure | Script | Shows | Reads | Made by |
|---|---|---|---|---|
| S1 | `figures/Figure_S01.jl` | ground-truth oscillators | `cache/truth` | `systems_truth.jl` |
| S2 | `figures/Figure_S02.jl` | validate by rollout, not one-step error | `cache/selfdrive/validation_criteria.jls` | `validation_criteria.jl` |
| S3 | `figures/Figure_S03.jl` | large kicks to coupled FHN: examples and all held-out kicks | `cache/cfhn_smooth/large_kicks` | `cfhn_smooth_regimes.jl` → `cfhn_large_kicks_data.jl` → `cfhn_large_kicks_models.jl <dmd\|esn\|lstm seed>` |
| S4 | `figures/Figure_S04.jl` | small kicks: relaxation and linear-model affordances | `cache/fig6` | `offcycle_v2.jl`, `slip_decays.jl`, `slip_rotation.jl` (← `slip_rotation_truth.jl`) |
| S5 | `figures/Figure_S05.jl` | how the phase-varying model of running deviations is built (one trial) | `cache/maus/angles.jls` | `maus_extract.jl`; model code in `maus_learning_all.jl` |
| S6 | `figures/Figure_S06.jl` | Floquet multipliers read out of fitted models | `cache/floquet` | `floquet_from_models.jl` |
| S7 | `figures/Figure_S07.jl` | identifying individuals: mean cycle against operator signature | `cache/fig5/gait.jls` | `fig5_gait_data.jl` |
| S8 | `figures/Figure_S08.jl` | imposing the limit cycle's structure on a linear model (lattice) | `cache/floquet`, `cache/fig5` | `figS6_slip_data.jl`, `floquet_lifted.jl` (← `slip_long_decays.jl`), `gait_lattice.jl`, `gait_lattice_null.jl`, `gait_lattice_speed.jl` (← `slip_kicked_walks.jl`) |
| S9 | `figures/Figure_S09.jl` | forecast baselines (SSMLearnPy) | `cache/ssm_baselines.npz`, `cache/fig_forecast_examples.npz` | `SSM_baselines.py` (← `ssm_fit_gait.py`), `export_forecast_examples.py` |
| S10 | `figures/Figure_S10.jl` | neural ODE on gait | `cache/gait_overparam_v2_*.jls` | `gait_overparam.jl` |
| S11 | `figures/Figure_S11.jl` | off-cycle relaxation | `cache/floquet` | `offcycle_v2.jl` |
| S12 | `figures/Figure_S12.jl` | SLIP return-map eigenvalues; piecewise-linear maps against the number of sectors (companion to Figure 7 E–F) | `cache/fig3` | `fig3_slip_maps.jl`, `slip_truth.jl` |

## Shared code

- `paper_style.jl` — CairoMakie theme, colour roles, `save_paper`.
- `common.jl` — paths (`cache_path`), gait data loading, Hankel/DMD helpers, oscillator simulators.
- `self_driving_benchmark.jl` — systems, regimes, model fits (DMD, ESN, neural ODE), `score`, `fit_template`.
- `dmd_tuning.jl`, `esn_tuning.jl` — tuned settings and DMD variants; `lstm_model.jl` — Winner et al. (2023) LSTM
  (Keras initialisation; `fit_lstm_winner_many` trains many independent per-trial models at once on the GPU).
- `fig5_gait_data.jl` — stride cutting, mean cycles, operator fits on gait; `heterogeneity.jl` — the H ratio.
- `cfhn_smooth_system.jl` — coupled FitzHugh–Nagumo (smooth regime); `slip_truth.jl` — SLIP walker (SLIP.jl env).
- `maus_lattice.jl` → `maus_horizon.jl` → `maus_floquet_models.jl` → `maus_learning.jl` → `maus_learning_all.jl` —
  running data, deviations, phase-varying linear model (`fit_ltp`), lattice and baselines.

## Analyses without a figure (results quoted in the text, or in progress)

- `gait_dimensionality.jl` — how many principal directions the whole signal and the departures from the average stride
  occupy in the 299 walking trials (quoted in the introduction) → `cache/fig5/dims.jls`.

- `gait_group_fits.jl` — individual against group fits on all 299 gait trials (DMD, ESN, LSTM; planned supplementary figure).
- `cohort_fits.jl` — individual against group fits on the simulated cohorts.
- `iv_running.jl` — least-squares against instrumental-variable section maps on the running trials, by training strides,
  and instrument strength (correlation of the section state one stride apart) for SLIP, running and walking; not in a
  figure (the SLIP comparison is Figure 7 E–F).
- `large_kicks_dmd_probe.jl`, `cfhn_density_probe.jl` — why DMD misses the coordination of coupled FHN (cycle harmonics,
  noise, training size, sampling density).
- `iv_gait_stride.jl`, `iv_gait_operator.jl`, `fig6_bias_correction.jl`, `dmd_noise_variants.jl`, `floquet_tuned.jl`,
  `coupling_benchmark.jl` — results reported in the text.

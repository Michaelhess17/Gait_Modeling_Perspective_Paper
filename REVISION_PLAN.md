# Revision plan — gait modeling perspective (updated 2026-10-18, evening)

## Direction (per Michael, latest)

**Main result: in self-driving (free-run) mode, the state-of-the-art nonlinear
models (neural ODEs, ESNs, SSMs, LSTM) fail on realistic biological oscillators
(FitzHugh–Nagumo, SLIP, real gait), while a linear Hankel-DMD model does not.**
The detailed DMD/non-normality analysis belongs in the companion paper, not here.
Coupling-direction analysis is shelved (results kept in `cache/coupling/`,
`summarize_coupling.jl`; see notes at the end).

Central benchmark: `self_driving_benchmark.jl` → `cache/selfdrive/`. Free-run
error at 1/3/5 cycles plus the 30-cycle fate (ok / decays with correct rhythm /
collapse / blow-up / wrong rhythm), with the mean-cycle template as reference.

Results (`Fig_self_driving`; pooled over both regimes). DMD and ESN use
hyperparameters chosen on training data only (`dmd_tuning.jl`, `esn_tuning.jl`:
hold out the last 2 training cycles); the neural ODE uses one fixed recipe (too
costly to tune per cell); SSMLearnPy uses its gait-tuned configuration.

| model | runs | ok | decays (right rhythm) | wrong rhythm | collapse | blow-up |
|### Floquet multipliers read from the models (`Fig_floquet_models`)

`floquet_from_models.jl`: chain of one-step Jacobians around each model's own
limit cycle (neural ODE, ESN); validated on the true systems (0.688 / 0.600).
DMD: consistent variational probe on data delay windows (cfhn only; the cycle
fills the observed plane for Stuart–Landau). Usable = settled cycle with period
within 10 % of the data.

| case (truth) | neural ODE (4 seeds) | ESN (8 seeds) | ESN (10 data sets) | DMD (88 settings) |
|---|---|---|---|---|
| cFHN, 30 cycles (0.69) | 0.0–0.29 | 0.21–0.80 | 0.17–0.58 | **0.46–1.05, median 0.68** |
| cFHN, 12 noisy (0.69) | 0.0–0.44 | 0.44–1.87 | 0.10–0.90 | 0.29–0.86, median 0.57 |
| SL, 30 cycles (0.60) | 0.77–1.0 (3/4) | 0.35–0.39 | 0.33–0.71 | n/a |
| SL, 12 noisy (0.60) | no usable cycle (4/4) | 0.10–0.27 | 0.03–0.65 | n/a |

- Caveats: the DMD probe uses windows taken from the data, which the nonlinear
  models do not get; quarter-cycle windows and the forward–backward fit were
  unstable and are excluded; the first DMD probe (newest-block kick) was
  inconsistent and gave 0.01–0.12.
- ESN and NODE fit/forecast well but the transverse multiplier is unconstrained:
  ESN spread 0.2–1.9 at equal fit; NODE reads ≈ 0 on relaxation dynamics and ≈ 1
  on the smooth oscillator.

### Floquet-map / piecewise demonstration (`Fig_floquet_maps`, replaces old Fig 4)

`floquet_maps_benchmark.jl` (+ `floquet_sl_control.jl`). Coupled FHN relaxation
oscillators (ε = 0.08, k = 0.02): one slow coordination multiplier |μ| = 0.688,
others ≈ 0; cycle-to-cycle process noise; 20 realisations per setting.

| estimator | 30 cycles (noise 0.25×) | 2000 cycles, noise 0 / 0.25 / 0.5 / 1 / 2 × deviation |
|---|---|---|
| phase-indexed next-cycle map (Revzen–Guckenheimer) | 0.59 [0.50–0.68] | 0.70 / 0.68 / 0.61 / 0.43 / 0.20 |
| Poincaré return map | 0.53 [0.47–0.71] | 0.65 / 0.66 / 0.65 / 0.62 / 0.52 |
| global map x(t+T) = A x(t), transverse eigenvalue | 0.52 [0.41–0.67] | 0.66 / 0.65 / 0.61 / 0.49 / 0.28 |
| piecewise linear, 8 sector-to-sector maps | 0.41 [0.32–0.45] | 0.67 / 0.40 / 0.13 / 0.01 / 0.00 |

- Data requirement: at clinical lengths the IQR spans most of the plausible
  range; estimates settle only after ~100–300 cycles.
- Noise: the textbook estimator is diluted toward 0 by measurement noise and
  more data do not help.
- Piecewise (sector-to-sector maps composed over one period): with clean data it
  recovers |μ| for both the smooth control (Stuart–Landau) and the relaxation
  pair; under small noise each sector map is attenuated and the attenuation
  multiplies, so finer partitions are worse (SL 0.95→0.57×, cFHN 0.77→0.03×
  truth from K = 4 to 64 at noise 0.25×), and more data do not help.
- Global raw-coordinate map: the directions the cycle occupies give unit
  eigenvalues; the next one tracks |μ| here only because the synchronous cycle
  spans 2 of 4 dimensions. A cycle that fills the observed space (Stuart–Landau,
  likely gait) leaves nothing to read.
- Earlier versions of this panel over-claimed (global map "stuck at 1",
  piecewise "fails without noise"); both were eigenvalue-extraction artefacts.
- Caveat for gait: a noise ratio of 1 ≈ 6 % of signal SD here. White marker
  noise in the gait recordings is ≈ 0.4 % of SD, so for gait the binding
  constraint is record length plus the unknown fraction of "deviation" that is
  non-white measurement error (soft tissue, marker placement).

---|---|---|---|---|---|---|
| ESN (tuned, 8 seeds) | 104 | 60 % | – | 12 % | 9 % | 19 % |
| neural ODE (4 seeds) | 52 | 42 % | – | 27 % | 19 % | 12 % |
| SSMLearnPy (2 degrees) | 26 | 62 % | – | 4 % | 23 % | 12 % |
| Hankel-DMD (tuned, deterministic) | 13 | **85 %** | 15 % | 0 | 0 | **0** |

Locomotion only (SLIP + 5 gait): tuned ESN 19/56 ok with 20 blow-ups; neural
ODE 1/28 ok; SSM 11/14; tuned DMD 5/7 ok, 7/7 right rhythm, 0 blow-ups.

- The first DMD pass was badly under-tuned (one P/10 delay window, clipping only
  |λ| > 1). Delay windows of P/4 to P with forward–backward fitting or
  unit-circle placement of persistent modes fixed it; FHN no longer collapses.
- Caveats for the text: DMD is one run per cell vs multiple seeds for the
  others; the neural ODE was not tuned; the FHN DMD waveform is visibly rounded
  even when its amplitude and period pass; the template still forecasts best.

---|---|---|---|---|
| ESN | 31/48 | **8/56** | 8 | **47** |
| neural ODE | 21/24 | **1/28** | 1 | 6 |
| SSMLearnPy | 5/12 | 11/14 | 11 | 0 |
| Hankel-DMD | 4/18 | 1/21 | **18** (amplitude decays) | **0** |
| template | 6/6 | 7/7 | 7 | 0 |

- ESN and neural ODE: work on textbook oscillators, fail on SLIP and gait
  (blow-up, wrong rhythm, collapse; outcome depends on seed).
- SSMLearnPy (tuned on gait): holds up on locomotion at one polynomial degree,
  fails at the other and on VdP. Fragile to hyperparameters, not uniformly
  failing; say so.
- DMD: never diverges; keeps the correct rhythm in 18/21 locomotion runs but the
  free run loses amplitude, and it collapses on relaxation oscillators (FHN).
- The template beats every model at forecasting; keep as the reference line.

---

## Earlier direction (coupling focus, superseded)

The perspective explains **why flexible nonlinear models struggle to reveal
locomotor dynamics and coordination from realistic data**, as context for our
DMD-based papers. Primary focus: **weaknesses in detecting coupling and
coordination** in nonlinear models (neural ODEs, ESNs, SSMs; LSTM "gait
signatures" as the field's reference point). Not about defending absolute
non-normality values.

Working thesis:

> Nonlinear models can reproduce walking kinematics, but with realistic data
> (8–14 strides, measurement and process noise, partial observation, drive) the
> structure they learn — rhythm, stability and, above all, who drives whom
> between limbs — is unconstrained by the data, changes with the random seed,
> and is often wrong in direction. A forecast that looks good is not evidence of
> the right structure. Linear delay-coordinate operators (Hankel-DMD) have their
> own limits, but they are deterministic, fit in closed form, and expose
> coupling structure directly. Where they get it right, and where they don't,
> is what this paper quantifies.

The last sentence is what the coupling benchmark (§3) must establish; the pilot
is mixed (below), so the claim is only as strong as those results.

---

## 1. Evidence status

| Claim | Evidence | Status |
|---|---|---|
| Realistic trials are short and mostly mean cycle | 8–14 strides per trial (LF median 8); fixed-period mean cycle explains a median 94 % of variance | numbers checked; figure todo |
| Forecast skill ≠ structure; SSM struggles | SSMLearnPy loses to a phase-aligned mean-cycle template at every horizon (0.28/0.38/0.62 vs 0.17/0.27/0.39; SSM worse in 70–80 % of trials). Hankel-DMD r = 10 ≈ SSM, so this is not a DMD win on forecasting. | `Fig_forecast_baselines` **done** |
| NODE outcome depends on training regime; field right only near the data | Single shot collapses or blows up; multiple shooting fits fully observed toy systems (sine R² 0.95); Van der Pol field R² 0.91 near data vs 0.40 across the box. Michael's Python pipeline (single-channel Takens + derivative, encoder-inferred IC, windows up to the full trial) diverged outright (best loss inf), consistent with partial observation being the killer. | `Fig_node_sim` **done** |
| NODE on real gait: equal fit, different dynamics | 3 trials × 6 seeds (35k params, loss 0.016–0.034): learned cycle 0.7× to 21× the stride period; 5 of 30 fits settle on cycles ≥ 3 strides long, 1 has no cycle; among stride-like cycles, |μ| spans 0.38–0.80 within one HF trial at equal loss. HF forecasts leave the hip–knee loop within ~1 s; most seeds lose to the template. | `Fig_node_gait` **done** |
| Nonlinear models can't recover coupling direction | Pilot (ideal, A→B): ESN forecasts the held-out cycle perfectly (error 0.00) but its asymmetry index is −0.56 to +0.34 across seeds (truth +1; 6/8 wrong sign). DMD has the right sign in 10/12 rank × delay settings (AI up to +0.68) but is too weak. **All models overestimate cross-talk ~12×** (CI ≈ 1 vs truth 0.08) on clean on-attractor data, likely because coupling only shows off the attractor. | full grid **running** |
| Why simulated success doesn't transfer | Calibrated ablation (coupled FHN, NODE): gait-level noise irrelevant; drive and partial observation fatal. The partial-obs condition lacks delay embedding (unfair) and should be redone. | needs fair partial-obs condition |
| ESN fails on Floquet structure | Recovers only the trivial multiplier even on simulated systems. Paper3 notes the estimate may come from a raw-data stride map rather than the ESN; the code needs checking. | fold into coupling benchmark |

### Caveats the paper must state (from this session and Paper3)
- In delay coordinates, non-normality reflects the dynamics *and* the
  measurement/embedding geometry. A truly normal 24-D system seen through 6
  channels reads Henrici 0.29 at r = 16, as gait does, and no normal model fits
  it either (`normality_necessity.jl`). Don't argue for DMD via absolute
  non-normality; argue via structure that can be checked against ground truth.
- DMD doesn't beat the template on forecasting either.
- DMD magnitudes depend on rank, delays, stabiliser and norm; directional and
  contrast quantities at fixed declared settings are what survived in Paper3 and
  the rodent project.

---

## 2. Proposed figures

1. **What realistic locomotion data look like.** Strides per trial, mean-cycle
   fraction, noise / non-harmonic power / drift levels (calibration cache).
   Defines the realistic regime used everywhere else. *todo*
2. **Forecast skill is not structure.** `Fig_forecast_baselines`. *done*
3. **Neural ODEs on toy systems.** `Fig_node_sim`. *done*
4. **Neural ODEs on real gait.** `Fig_node_gait`. *done*
5. **Coupling benchmark (central).** Directed-coupled Van der Pol pair; one
   model-agnostic perturbation readout for every model; asymmetry and cross-talk
   vs truth across models × conditions (clean, short, noisy, process noise,
   partial obs, gait-like); seed or setting spread; forecast skill vs coupling
   accuracy. *running*
6. **Why realistic data break the models.** Calibrated ablation, redone with a
   fair partial-observation condition. *todo*
7. **Box/Table: practices.** Baselines (template), seed and setting spread,
   ground-truth coupling checks, declared DMD settings, mean-cycle surrogates.

## 3. Coupling benchmark — design (`coupling_benchmark.jl`)

- System: ẍ_A − μ(1−x_A²)ẋ_A + x_A = k_AB x_B; ẍ_B − … + 1.05² x_B = k_BA x_A
  (μ = 1, dt = 0.05, ~133 samples/cycle). Configs: none, A→B, B→A, bi (k = 0.2);
  A→B strength 0.05/0.1/0.2/0.4. Coupled cases lock 1:1; uncoupled drifts.
- Readout: kick position j by 0.1 SD at 20 phases; RMS response of position i
  over half a cycle. AI = (R_BA − R_AB)/(R_BA + R_AB); CI = cross/self response.
  Truth from the noise-free true system.
- Conditions: ideal (50 cycles, clean), short (12), noisy (5 % meas.), process
  (σp = 0.1), partial (positions only), gaitlike (12 cycles, 0.4 % meas., σp
  0.05), realistic (gaitlike + positions only + 1 % noise).
- Models: DMD (3 delays × 4 ranks), ESN (8 seeds), NODE (4 seeds, full obs only
  for now). Todo: LSTM, SSMLearnPy, NODE on delay coordinates for partial obs.

## 4. Decisions still open
1. Include rodent results as an example? (unpublished project)
2. Venue/length.
3. Should LSTM ("gait signatures") be in the benchmark? Recommended, since it's
   the field's reference nonlinear model.


### Retraction (2026-10-20): "linear models can't represent radial relaxation"

An earlier note said Stuart–Landau needs nonlinear observables because a linear
operator holds a perturbed amplitude fixed. That is wrong as stated: a signal
(r* + Δe^(−γt))cos ωt is a sum of exponentials, so a delay-coordinate linear
model can represent it. Hankel-DMD trained on decay data (perturbations ≥ ~35 %
of the radius) with a delay window ≥ 1 cycle reproduces the 0.6/cycle relaxation
(`dmd_offcycle_decays.jl`). The earlier "frozen radius" came from (i) training
data with only ~8 % noise-level deviations, (ii) delay windows ≤ ¾ cycle, and
(iii) hyperparameter selection by on-cycle validation, which cannot see the
transverse direction. The Stuart–Landau DMD multipliers from the single-trajectory
runs (0.99, 0.84) therefore show that on-cycle data plus on-cycle validation do not
identify the transverse direction, not a limit of the model class.

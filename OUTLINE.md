# Outline: revised perspective draft

Working title options

- *Transients, not stability: what a data-driven model of walking must do, and why linear operator models do it best*
- *Modelling gait as a perpetually perturbed oscillator: transient dynamics, model requirements, and linear operator models*
- *Beyond the mean gait: transient structure, learnability, and linear models of locomotor dynamics*

Format: perspective (e.g., J. R. Soc. Interface / PLOS Comput Biol perspective), ~6 main figures plus a supplement.

Marks used below:

- **[have]**: the result exists in this repo (the source file is named).
- **[new]**: needs new analysis.
- **[pending]**: running now.
- **[cite✓]**: a reference checked during drafting.
- **[cite?]**: a reference from memory that needs checking before use.

Structure (restructured 2026-11-03; the mapping from the previous outline is at the end): the target (§2) → what a model must do to study it (§3) → which candidate does it (§4) → what that model shows on real gait, and where the data stop (§5).

---

## Abstract (sketch, 5 sentences)

1. Gait dynamics are usually summarised by asymptotic stability (Floquet multipliers), and data-driven dynamical models (neural ODEs, reservoir computers, sparse regression, spectral-submanifold reductions) are increasingly fitted to walking.
2. But walking never reaches the asymptotic regime: it is perturbed every stride and corrected by noisy feedback, so what governs it is the transient response over the next few strides, which eigenvalues of non-normal return maps miss and which short, noisy records estimate poorly.
3. Studying transients asks specific things of a model. Above all it must predict: run on its own from the current state, it should forecast held-out strides, on the cycle and after perturbations, with skill beyond reference forecasts that need no dynamics (the mean cycle, the last stride). It should also keep the right cycle over long rollouts, give one answer per data set, behave predictably under measurement noise, and work from modest data.
4. Against these requirements, on realistic oscillators, SLIP and human gait, dynamical models forecast far beyond the reference where the data hold predictable dynamics, and barely beyond it on noise-driven data. Expressive models are sometimes the best forecasters but often blow up, settle on the wrong attractor or depend on the seed. Hankel-DMD, linear in delay coordinates, forecasts consistently (close to the best model in most cases and never far below the reference), keeps the right cycle in 92% of cases, gives one convex answer, has an analysable and correctable bias and fits in seconds, with known limits (a neutral cycle amplitude, a need for transients).
5. No model recovers transverse dynamics that passive on-cycle data do not contain, and 15-s gait trials do not identify stride-to-stride dynamics, so we recommend perturbation experiments and reporting finite-time response with its uncertainty.

---

## 1. Introduction

**¶1. Motivation.** Walking is regulated stride by stride, and understanding how the nervous system and the body jointly produce, and correct, each stride is central to explaining gait impairment and designing interventions.
- Clinical hook: stroke, Parkinson's disease, aging; fall risk; individual-specific responses to exoskeletons and rehabilitation.

**¶2. State of the field: mechanistic templates.** The oldest route to gait dynamics is mechanistic: low-dimensional templates and detailed musculoskeletal simulations that generate walking from physical principles.
- Templates and anchors: Full & Koditschek 1999 [cite?]. SLIP: Blickhan 1989; Geyer et al. 2006 [cite?]. Passive dynamic walking: McGeer 1990; Garcia et al. 1998 [cite?]. Holmes et al. 2006 SIAM Review [cite?].
- Neural control: CPG and reflex controllers (Taga 1991; Geyer & Herr 2010; Ijspeert 2008 review) [cite?].
- Predictive musculoskeletal simulation (Falisse et al. 2019) [cite?]; deep RL controllers (Song et al. 2021) [cite?].
- Limitation: these reproduce typical gait but are hard to fit to an individual, and they often fail to predict changes in kinematics (point made in Winner et al. 2023 [cite✓]).

**¶3. State of the field: stability and variability metrics.** A second tradition summarizes gait dynamics with stability and variability statistics computed directly from kinematics, not with a generative model.
- Floquet multipliers: Hurmuzlu & Basdogan 1994 [cite?]; Dingwell & Kang 2007 [cite✓]. Local divergence exponents: Dingwell & Cusumano 2000 [cite?]. Margins of stability: Hof 2005 [cite?]. Long-range correlations: Hausdorff 1995 [cite?].
- Known fragility: maximum Floquet multipliers depend on the number of strides and the state-space composition, and are biased by noise (Riva et al. 2013 [cite✓]; Bruijn et al. 2013 review [cite?]).

**¶4. State of the field: data-driven models.** A third, newer tradition fits dynamical models to gait data. The methods range from linear step-to-step maps to deep sequence models, and each trades interpretability for expressiveness differently.
- Step-to-step linear maps that infer control from variability: Wang & Srinivasan 2014 [cite?]; Seethapathi & Srinivasan 2019 eLife [cite✓]. Goal-equivalent manifolds: Dingwell & Cusumano [cite?].
- Phase-based models: Revzen & Guckenheimer 2008/2012 [cite?]; phase-varying models of exoskeleton response, Rosenberg et al. 2020 [cite?]; Hybrid-SINDy for centre-of-mass dynamics, Rosenberg, Proctor & Steele 2024 Sci Rep [cite✓].
- Neural models: RNN gait signatures, Winner et al. 2023 PLOS Comput Biol [cite✓], extended 2024 Sci Rep [cite✓]; modulated neural ODEs evaluated on human walking trajectories (MoNODE, NeurIPS 2023) [cite✓].
- Koopman/DMD in gait: Fujii et al. 2019 Sci Rep (DMD for coordinative structures) [cite✓]; Koopman model of whole-body angular momentum with periodic regularization (NSF PAR 10613388) [cite✓, get the full reference].

**¶5. The gap.** The stability tradition and most model-based work aim at the same quantity, asymptotic stability, in a system that never reaches the asymptotic regime. And there is little systematic evidence on whether expressive data-driven models predict walking beyond its average cycle, or recover the dynamics around it. Methods that work convincingly on simulated benchmarks and fluid flows are often assumed to transfer.

**¶6. This perspective.** We (i) argue that for a gait perturbed and noisily corrected every stride, the right target is the transient, finite-time response, not asymptotic stability (§2); (ii) derive what a data-driven model must do to study that response, above all predict it, and why gait data make it hard (§3); (iii) test expressive and linear models against these requirements and find that linear models in Hankel (delay) coordinates meet them best (§4); and (iv) show what linear models reveal on real gait and where 15-s trials stop every model (§5).

---

## 2. Asymptotic stability is the wrong target for gait

**¶1. Why Floquet multipliers became the standard.** Floquet multipliers answer a precise question: how fast, asymptotically, a perturbation shrinks per cycle. That made them the natural stability measure for gait.
- Hurmuzlu & Basdogan 1994 [cite?]; Dingwell & Kang 2007 [cite✓].

**¶2. Walking never reaches the asymptotic regime.** Gait is perturbed every stride by intrinsic noise and by the environment, so what governs behaviour is the response over the next one to three strides, the transient. Asymptotic multipliers can miss it entirely.

**¶3. In a noisily controlled system, variability is made of transients.** When every stride is kicked and every stride is corrected, the deviation at any stride is the sum of the decaying responses to all earlier kicks. Stride-to-stride variability is the transient response, sampled continuously, and how deviations are corrected reveals the control strategy.
- Seethapathi & Srinivasan 2019 [cite✓]; Wang & Srinivasan 2014 [cite?]; goal-equivalent manifolds, Dingwell & Cusumano [cite?]; uncontrolled manifold, Scholz & Schöner 1999 [cite?]; minimal intervention, Todorov & Jordan 2002 [cite?].

**¶4. Eigenvalues miss the transient: return maps of oscillators are inherently non-normal.** Even for a textbook Van der Pol oscillator, the monodromy matrix and stride-to-stride return map are non-normal: their eigenvectors are far from orthogonal, and within-cycle variational dynamics amplify perturbations transiently before they decay. Eigenvalues are not the full story.
- Trefethen & Embree 2005; Trefethen et al. 1993 "Hydrodynamic stability without eigenvalues" [cite?].
- Bridge to the companion paper; keep this brief.

**¶5. Multipliers are also data-hungry and noise-fragile.** Estimating multipliers from return maps needs many cycles and low noise. Piecewise-linear phase-sector maps compound the problem, because noise attenuation multiplies across sectors.
- Revzen & Guckenheimer [cite?]; Riva et al. 2013 [cite✓].
- Our maps: phase map 0.59 [0.50–0.68] at 30 cycles against a truth of 0.69; the 8-sector piecewise map gives 0.41 and collapses to about 0 at noise ≥ 1× **[have: on coupled FHN]**. Fig 2 now uses kicked SLIP walks **[have: Fig3_floquet_maps.jl, fig3_slip_maps.jl]**: replace these numbers with the SLIP ones from cache/fig3/slip_maps.jls.
- **The section-map bias is correctable, because the map is linear [have: Fig 2B, C].** Least squares treats the current stride as exact, so measurement noise pulls ρ̂ toward stability, and more strides do not help (open loop 0.95 → 0.88 / 0.75 / 0.47 at noise 0.25 / 0.5 / 1×, 250 steps). Instrumental variables, with the previous stride as the instrument and no knowledge of the noise, recover the truth: 0.94 / 0.95 / 0.97 open loop, 0.87 / 0.88 / 0.91 with speed feedback (truth 0.872). Below about 30 strides the instrument is weak and IV scatters and overshoots.

**¶6. A better target: finite-time, stride-scale response.** We propose characterizing gait by its finite-time response operators: how deviations at one stride propagate, amplify and redistribute across joints over the next few strides. These are quantities that linear models expose directly, and a model that predicts the next few strides from the current state, including after a perturbation, is a model of exactly this response.

**Figure 1: Eigenvalues are not the whole story** **[have: Fig4_nonnormality.jl; ground truth only, systems_truth.jl and slip_truth.jl: Van der Pol μ sweep, Stuart–Landau, coupled FHN, SLIP open loop and with speed feedback]**.
- (A) Transverse distance from the cycle after a perturbation against time, with the multiplier's prediction |μ|^τ.
- (B) Peak within-cycle transverse amplification against the transverse multiplier; shaded, the multipliers reported for human walking (≈0.3–0.8).
- (C) Stride to stride: worst-case growth of the n-stride transverse map against the eigenvalue prediction ρⁿ.
- (D) Deviation from the unperturbed trajectory after n cycles, including lasting phase shifts; the eigenvalue prediction is 1.

**Figure 2: Data-driven Floquet maps of SLIP walking are data-hungry and noise-fragile** **[have: Fig3_floquet_maps.jl, fig3_slip_maps.jl; kicked SLIP walks, truth ρ = 0.95 open loop]**.
- (A) Eigenvalues of the section (VLO) return map: 30 steps with noise 0.25×, against 2000 noise-free steps.
- (B) Spectral radius against record length, noise-free and with noise 0.25×: more strides reduce scatter, not the noise bias; the IV section map removes it.
- (C) Against measurement noise, 250 steps: section map by least squares and by IV, phase-indexed maps, 8-sector piecewise-linear maps.
- (D) Piecewise-linear maps: bias grows with the number of sectors composed over one step.

---

## 3. What a model of gait transients must do

**¶1. From target to requirements.** A model is useful for gait to the extent that it predicts: from the current state, what the next strides will do, including after a perturbation. Studying transients adds requirements that short-horizon prediction alone does not guarantee. Six in all:
- **(R1) Predictive skill in autonomous rollouts.** Run from the current state with no data fed back, the model forecasts held-out data over the next few strides, on the cycle and after perturbations.
  - Skill is measured against reference forecasts that need no dynamics, the mean cycle at the current phase and the last stride repeated, as weather forecasts are scored against climatology and persistence (skill = 1 − model error / reference error). The references are not candidate models (they have no state and cannot be started off the cycle); they define zero skill.
  - Skill is reported on the full signal and on the deviations from the mean cycle, because the deviations are where the dynamics are.
  - Errors are validation errors: computed on held-out data (later cycles of a trial; held-out decays or realisations), by autonomous rollout, not one-step (teacher-forced) error **[have: validation_criteria.jl, Fig S7]**.
    - Every setting of the benchmark's tuning grids (DMD 60; ESN 36 × 3 seeds), in all 13 cells, fitted without its last two training cycles and scored on them by four criteria, then refitted and tested.
    - Rollout errors rank settings by their test forecast error; one-step errors do not. Median rank correlation across settings: DMD 0.87 (one held-out rollout) and 0.88 (multistart: 1-cycle rollouts from 5 starts) against 0.40 (in-sample one-step) and 0.38 (held-out one-step); ESN 0.71 and 0.75 against −0.19 and 0.17.
    - As a selection rule for the ESN, in-sample one-step error picks blow-ups (right cycle kept in 2 of 13 cells, median test skill −4.2); held-out one-step 7 of 13 (−0.11, with one choice at −72); one held-out rollout 5 of 13 (+0.17, occasional catastrophes); multistart rollout 9 of 13 (+0.16, worst −2.7), as often as the setting best on the test data itself (9 of 13, +0.53). For DMD every criterion chooses passably (8–11 of 13; median skill −0.22 to +0.16).
    - The setting best on the test forecast keeps the cycle in only 9 of 13 cells for both models: short-horizon skill and long-run behaviour are separate requirements (R1, R2).
  - Reference levels on gait, held-out error at 0.7 / 2.7 / 4.7 s lead **[have: FigS1]**: mean cycle 0.17/0.27/0.39; last stride repeated 0.20/0.30/0.42; Hankel-DMD (rank 70, ridge 1e-4, stabilised; the companion paper's settings) 0.21/0.35/0.46 (rank 10: 0.27/0.40/0.54; a 0.6–1.5-s window: 0.19/0.30/0.42–0.44 **[window test not in the figure]**); spectral-submanifold model 0.28/0.38/0.62.
- **(R2) Faithful long-run behaviour.** Over long rollouts it keeps the right cycle (no blow-up, wrong period or collapse) and returns to it after a perturbation: evidence that it learned the vector field around the cycle. Short-horizon skill does not guarantee this (§4 ¶3).
- **(R3) One answer.** The same data give the same model and the same stability readout across seeds and nearby hyperparameters, so differences between people are not fitting noise.
- **(R4) Predictable behaviour under measurement noise.** Its estimates are insensitive to kinematic noise, or biased in a way that can be analysed and corrected.
- **(R5) Modest data needs.** It works from tens of strides and a few channels, and shows when the data cannot support an answer.
- **(R6) Cheap and inspectable.** It fits fast enough for cross-validation and bootstrap, with spectra and modes that can be checked.

Why these requirements are hard to meet on gait (learnability):

**¶2. Where data-driven dynamics succeed.** Neural ODEs, SINDy, reservoir computers and DMD have their clearest successes on systems with fixed governing equations, high signal-to-noise ratio, abundant snapshots, and dynamics that are observed through their transients.
- Cylinder wake (Bagheri 2013 [cite?]), Lorenz/KS with reservoirs (Pathak et al. 2018 [cite?]), SINDy (Brunton et al. 2016 [cite?]), neural ODEs (Chen et al. 2018 [cite?]).
- In fluids, start-up transients and parameter sweeps supply off-attractor data cheaply.

**¶3. Walking is a closed loop.** In walking, mechanics and neural control form one feedback loop. Passive recordings of the loop cannot separate plant from controller, a classic result in closed-loop system identification.
- Forssell & Ljung 1999 [cite?].
- Standing balance needs deliberate perturbations to separate plant and feedback (van der Kooij, Pasma, Schouten and colleagues; closed-loop system identification) [cite✓].
- Our coupling test: no model, linear or nonlinear, recovers coupling direction from passive data **[have: coupling_benchmark.jl, cache/coupling]**. One sentence plus a supplement pointer.

**¶4. Feedback hides the transverse dynamics (R1, R2, R5).** Because the controller suppresses deviations, walking data stay close to the limit cycle. The transverse directions that encode stability and control are excited only weakly, by intrinsic noise.
- The mean cycle explains about 94% of gait variance **[have]**.

**¶5. Short, hybrid, partially observed, nonstationary (R1, R4, R5).** Gait datasets add further obstacles: tens of strides per trial, a few kinematic channels standing in for a high-dimensional neuromechanical state, discrete contact events, and slow drift in speed and posture.
- Deviations are 1.4–4.7% of a trial's variance (median by group, Fig 6B), so measurement noise is not small against the signal a transient analysis uses.
- On passive, noise-driven data the forecast skill any model can reach beyond the mean cycle is therefore small: the deviations are barely predictable from the current state (§5 ¶4). Prediction is tested best where the data hold predictable deviations, after perturbations.

**¶6. Consequence: on-cycle data identify the cycle, not the dynamics around it (R2, R3).** Many models fit on-cycle data equally well yet disagree about everything off the cycle, so training loss cannot select among them.
- Gait neural-ODE fits reach similar training loss with learned periods from 0.7× to 21× the stride and different multipliers **[have: Fig_node_gait, gait_overparam.jl]**.
- Held-out forecasts (5 s after the 10-s training window) do not pin the dynamics either **[have: Fig 3B, gait_overparam_validation.jl]**. On the LF trial the best held-out forecasts (error 0.24–0.49) come from stride-like cycles, with multipliers 0.54–0.73, and the wrong-rhythm fit is clearly worse (1.52). On the AB trial the best held-out forecasts (0.29–0.86) come from fits whose cycles are 9–15× the stride period. On the HF trial no fit forecasts better than the mean (1.17–1.65).
- Floquet multipliers read from models trained on on-cycle data are unconstrained: Stuart–Landau neural ODE ≈ 1.0 and DMD ≈ 0.99 against a truth of 0.60 **[have: floquet_tuned.jl]**.

**¶7. Transient-rich data make the problem learnable, given enough of it (R1, R5).** When training data contain decays from perturbed states, transverse relaxation becomes identifiable. Even then, recovering it takes many transients and a delay window long enough to span the decay. Nothing is free.
- Hankel-DMD relaxation rate converges to the truth as decays increase (coupled FHN from about 30 decays; Stuart–Landau needs at least 1.5-cycle windows and about 300 decays) **[have: Fig_dmd_datasize, dmd_delay_sweep.jl]**.
- Experimental implication: perturbation paradigms (treadmill perturbations, split-belt, tripping) are the gait analogue of start-up transients.

**Figure 3: Learnability** **[have: Fig1_learnability.jl]**.
- (A) What the data cover: coupled FHN, the same 30-cycle budget as one long on-cycle trajectory or as 10 decays × 3 cycles from perturbed starts; on-cycle data barely sample the slow transverse direction.
- (B) Neural ODEs fitted to gait: learned period and multiplier against held-out forecast error **[have]**. Trials 36 and 90 (earlier runs, no held-out forecast cached) are left out until refitted.
- (C) DMD relaxation rate against amount of transient data **[have]**.
- (D, optional) Off-cycle relaxation of all models at matched data **[pending: ESN/NODE retune at 100 decays]**.

**Figure S7: Validate by rollout, not by one-step error (R1)** **[have: FigS7_validation.jl, validation_criteria.jl]**.
- (A) Rank correlation across settings between each validation error (in-sample one-step, held-out one-step, one held-out rollout, multistart rollout) and the test forecast error, per cell, DMD and ESN.
- (B) Test forecast skill of the setting each criterion chooses, with the number of cells where that choice keeps the cycle.

---

## 4. Candidate models against the requirements: linear models in Hankel space fit best

**¶1. Candidates and test design.** Expressive models: neural ODE, ESN, LSTM?, spectral-submanifold model (SSMLearn). Linear: Hankel-DMD, a linear operator on delay (Hankel) coordinates. Each is tuned on training data only with wide hyperparameter searches and tested by autonomous rollouts on held-out data. DMD and ESN hyperparameters are themselves chosen by rollout validation: fit on all but the last two training cycles, free-run, and pick by the error over those two cycles.
- Systems: coupled FitzHugh–Nagumo, coupled Van der Pol, Stuart–Landau, SLIP and human gait.
- Regimes: varied data length and noise level.
- **[have: self_driving_benchmark.jl, dmd_tuning.jl, esn_tuning.jl]**

**¶2. R1, predictive skill.** Where the data hold predictable dynamics, the dynamical models forecast far beyond the reference; where they are noise-driven, every model's skill is near zero, and what separates the models is whether they stay near the reference or fail.
- First held-out cycle, skill against the mean cycle (Fig 4C) **[have: Fig2_self_driving.jl printout]**. Long regime (30 clean training cycles): DMD +0.59 to +1.00 and ESN +0.68 to +1.00 on all four systems; neural ODE −3.5 to −0.08; SSM −73 to +0.40. Realistic regime (12 cycles, 1% measurement noise, process noise): DMD −0.43 to +0.17 on all nine cells; the ESN beats DMD on 5 of 9 (up to +0.42 on a gait trial) but falls below −1 on 3 (down to −13); neural ODE negative on all nine; SSM −4.5 to 0.
- Gait, the 5 s after a 10-s training window (Fig S1): DMD skill −0.24 / −0.30 / −0.18 at 0.7 / 2.7 / 4.7 s against the mean cycle; SSM −0.65 / −0.41 / −0.59. With noise-driven deviations, no model forecasts beyond the mean cycle here.
- Off the cycle: relaxation of decay-trained models, the same 10 decays for every model, from 4 off-cycle starts, over the first 4 cycles **[have: Fig 5, fig6_relaxation_data.jl]**. Deviation is measured to the curve through the reference's samples; for a model, from its own unperturbed run within ±½ cycle of the same time (scoring changed 2026-11-03, see below). Mean |model − truth| over starts (range over seeds):
  - Coupled FHN (truth 0.69 per cycle from every start): DMD 0.10 (0.77–0.81, too slow); ESN 0.14–0.18 over 4 seeds (0.80–0.87, too slow); neural ODE 0.19–0.32 over 3 seeds (0.27–0.59, too fast; cached runs, provisional reference).
  - SLIP, centre-of-mass state (truth 0.67–0.90 across starts): DMD 0.06 (0.83–0.86, nearly flat across starts: it misses the fast start, truth 0.67); ESN 0.11–0.23; neural ODE 0.05–0.13 (cached runs, provisional reference).
  - Stuart–Landau (dropped from the figure; truth 0.58): DMD 0.88, reproducibly wrong at this data size (it needs about 300 decays, Fig 3C); ESN 0.46–0.48; neural ODE 0.49–0.54.
  - **Why the scoring changed [have: fig6_relaxation_data.jl header].** (1) Distance to the nearest *sample* of the reference cycle has a floor: on coupled FHN consecutive cycle samples are up to 0.53 apart during the jumps, against deviations of 0.12–0.28, which made the truth's rate look start-dependent (0.70–0.84) and produced the 'backward' trends. Measured to the curve through the same samples (no denser data), the truth relaxes at 0.69 from every start. (2) Each model's cycle sits 0.13–0.16 from the true cycle, as far as the perturbation, so a model's perturbed and unperturbed runs both keep moving onto its own cycle, and DMD's waveform drifts slowly (higher harmonics ≈ 0.97–0.98 per cycle). The model's deviation is therefore measured from its own unperturbed run at the same time (±½ cycle, so phase shifts do not count), which cancels what the model does without the perturbation. Measuring from its settled attractor instead charges DMD for its drift (coupled FHN 0.18) and reorders the ESN seeds (0.08–0.16), so the number depends on the reference; the same-time comparison is the one that isolates the response to the perturbation. SLIP was not affected by (1) (spacing ≤ 0.16, deviations 0.37–0.51). The perturbation size is unchanged. Old numbers (nearest-sample scoring): coupled FHN DMD 0.07, ESN 0.16–0.28, neural ODE 0.15–0.32; SLIP DMD 0.04.
- Deviations one stride ahead, out of trial (Fig S6D): kicked SLIP walks R² 0.53 for the lattice model and 0.25 for a delay DMD on the deviations, against the mean stride; human gait about 0 for every linear model (§5 ¶4).
- Reading: prediction is where a model earns its use, and testing it needs data with predictable deviations, clean or perturbed transients. On passive, noise-driven data the attainable skill is small for any model, so there the comparison is about reliability: DMD is never far from the reference, while the ESN is sometimes the best forecaster and sometimes catastrophic.

**¶3. R2, long-run behaviour.** Hankel-DMD sustains the correct cycle in 92% of system–regime cells and never blows up. The expressive models succeed in 42–62% and blow up or settle on the wrong attractor in 12–19% **[have: Fig_self_driving]**.
- The failure modes differ in kind. Expressive models fail by blowing up, by settling on a wrong period, or by collapsing to a fixed point. DMD fails by decaying slowly, which is benign and predictable.
- Stability by construction: constraining eigenvalues (|λ| ≤ 1) guarantees bounded free runs, so the worst-case linear model is a slowly decaying one, not an exploding or wrong one **[have: self-driving fates]**.
- Short-horizon skill does not guarantee the right attractor. Among neural ODEs fitted to one able-bodied trial, the fit with the lowest held-out rollout error at 3 s settles onto a cycle 15× the stride period; on another trial, held-out error does flag the wrong-rhythm fit (1.05 against 0.24–0.40) that training loss does not **[have: gait_overparam_validation.jl]**.
- Models that forecast well can still get the multipliers wrong. Read out of fitted models, Floquet multipliers vary widely across model classes, seeds and hyperparameters, even among models that forecast similarly: neural ODE on coupled FHN 0.01 against a truth of 0.69; ESN 0.64 [0.38–0.86]; DMD 0.71 [0.66–0.79] **[have: floquet_tuned.jl]**. Honest caveat: on noisy Stuart–Landau the ESN is closest (0.57 against 0.60).
- **Multipliers of each model's own cycle (SLIP; coupled FHN's transverse multiplier is real) [have: Fig 5C, fig6_relaxation_data.jl].** The SLIP step map is a complex pair, 0.769 ∠ 13.6° per step. DMD's operator, read in the cycle frame (rotation from the nearest harmonic), has an eigenvalue at 0.763 ∠ 13.9° next to the 8th harmonic, but also a blended one at 0.827 ∠ 7.6° next to the 1st with almost the same ResDMD residual (0.052 against 0.050), so nothing in plain DMD marks which one is the Floquet sideband; the lattice-constrained fit picks 0.771 ∠ 14.6° without knowing the truth (Fig S6). The ESN's own cycles (monodromy of the closed-loop Jacobians over one period) contract faster than the truth: leading multipliers 0.36–0.64, real for three seeds, 0.48 ∠ ±21.5° for one. The neural ODE needs a refit to be linearised. (The earlier 2×2 section map read from the runs gave real eigenvalues for every fit and was not robust to the section state; dropped.)

**¶4. R3, one answer.** The DMD fit is a least-squares problem with a unique solution for each hyperparameter setting: no seeds, no local minima, and results that vary smoothly with rank and delay. Expressive models give different answers at equal loss (§3 ¶6).
- Honest reading: over the first 4 cycles DMD is the closest on coupled FHN (0.10 against the ESN's 0.14–0.18), though every model is off by at least 0.1 there, and as close as the best seeds on SLIP (0.06 against 0.05–0.23), with no seed spread in either. Over longer horizons it does not fully relax (¶10). The ESN's answer depends on the seed on both systems; the neural ODE relaxes too fast on coupled FHN. On these toy systems the seed spread is moderate; the large reproducibility failure is on gait (Fig 3B). DMD's advantage is that a wrong answer is the same wrong answer every time, and converges visibly with data.

**¶5. R4, measurement noise: the linear bias is analysable and correctable; the expressive models' is not.** In linear models we know how estimation errors bias the result and can correct it. Least squares treats the predictors X as noise-free and the targets Y as noisy; with measurement noise this pulls eigenvalues inward. Instrumental variables remove the bias on the SLIP return map (Fig 2B, C). Forward–backward and total-least-squares DMD are the corrections usually recommended, and optimized/bagged DMD adds uncertainty estimates.
- Dawson et al. 2016 [cite?]; Hemati et al. 2017 [cite?]; Askham & Kutz 2018 [cite?]; Sashidhar & Kutz 2022 [cite?].
- **The correction must match how the data were generated [have: fig6_bias_correction.jl; one or two sentences].** On stationary, noise-driven data (unperturbed walking), forward–backward eigenvalues come in reciprocal pairs λ, 1/λ (exactly in the long-record limit), so FB cannot report a decay. On the SLIP map it returns |λ| = 1.000 at every noise level, even without noise, while keeping the rotation per stride (48.4° against 48.6°). TLS is only partly right (open loop 0.91–0.96; with feedback 0.90 → 0.61 as noise rises). FB is designed for deterministic transients observed with noise, not for walking.
- Honest framing for Hankel-DMD: with rank-truncated Hankel matrices, plain least squares was already nearly unbiased up to 20% measurement noise on coupled FHN decays **[have: dmd_noise_variants.jl, noise_models.jl]**. On real gait, IV on the companion operator changed transient growth little and lowered it rather than raising it (⅓-stride window AB 0.27 → 0.21, LF 0.60 → 0.52; ordering unchanged) **[have: iv_gait_operator.jl]**. So the companion's transient growth is not deflated by measurement noise. For gait the correction is a safeguard, not a result, and it is not a comparison point in the paper.
- Expressive models are biased by construction, and corrections are rare. Neural ODEs, LSTMs and reservoirs trained by one-step or shooting loss inherit the same errors-in-variables bias, with no closed-form handle on it. Explicit corrections exist mostly for sparse regression.
  - Simultaneous denoising for SINDy, Kaheman, Brunton & Kutz 2020 [cite✓]; neural ODEs with an implicit denoising network, Goyal & Benner 2023 [cite✓]; time-reversal-symmetry penalty, Huh et al. 2020 NeurIPS [cite✓] (closest analogue to forward–backward consistency).
  - No total-least-squares or errors-in-variables neural ODE found (state as "to our knowledge").
- Exception: on noisy Stuart–Landau the ESN relaxes more reliably than DMD (¶10).

**¶6. R5, data needs.** Hankel-DMD's relaxation rate converges visibly as decays are added (Fig 3C, §3 ¶7): coupled FHN from about 30 decays, Stuart–Landau only with ≥ 1.5-cycle windows and about 300. Whether the expressive models need fewer at matched data is Fig 3D **[pending]**. On real 15-s gait trials no model has enough (§5 ¶4).

**¶7. R6, cost and inspection.** Fits take seconds, which makes cross-validation, bootstrap over trials and subjects, and wide hyperparameter sweeps routine. A single neural-ODE fit takes hours **[have: timing from runs]**. Linear spectra can be verified with residuals and constrained by theory (§5 ¶5).

**¶8. Implementation is not the explanation.** These failures persist with careful training: multiple shooting, adjoint gradients, RK4 integration, and train-only selection over wide searches. Common pitfalls make them worse (Box 1).
- **Box 1 / supplement: neural-ODE pitfalls.** Forward Euler at the sample step spirals outward on oscillations; an output nonlinearity biases the vector field; very short shooting windows; latent encoder/decoder non-identifiability; summed weight decay; selection on minibatch loss. **[new, optional: a controlled degradation test on Stuart–Landau, about 1 h]**

**¶9. Scorecard [new: render as Table 1, from the results above].** Hankel-DMD is the only candidate that meets R1–R6 together on these benchmarks. It does not win every cell (¶10).
- R1, first-cycle forecast skill against the mean cycle (Fig 4C): DMD and ESN near-exact on clean data (+0.6 to +1.0); on noisy data DMD −0.43 to +0.17, ESN −13 to +0.42, neural ODE and SSM mostly negative. Gait over 5 s (Fig S1): DMD −0.18 to −0.30, SSM −0.41 to −0.65. Off the cycle (Fig 5): DMD closest on coupled FHN (0.10 against 0.14–0.32) and as close as the best seeds on SLIP (0.06 against 0.05–0.23).
- R2: DMD keeps the correct cycle in 92% of cells and never blows up; expressive models 42–62%, with blow-up or a wrong attractor in 12–19% (Fig 4); beyond about 8 steps DMD's off-cycle deviation plateaus (¶10), where the ESN returns to its cycle.
- R3: DMD gives one answer; the ESN depends on the seed on both systems; the neural ODE is consistent on SLIP, erratic on coupled FHN, and its equal-loss fits to gait disagree (periods 0.7–21×).
- R4: DMD nearly unbiased up to 20% noise on coupled FHN decays, and linear-map bias correctable (IV, Fig 2B, C); no handle for the expressive models; exception on noisy Stuart–Landau.
- R5: DMD converges visibly with transients (Fig 3C); expressive models at matched data pending (Fig 3D).
- R6: seconds against hours per fit.

**¶10. Where linear models fall short.**
- Linearity is not a free lunch. DMD needs transient-rich data and long enough delay windows. On noise-driven data its forecasts stay near the reference, and the best ESN settings forecast better on 5 of 9 cells (¶2). Its settings matter on Stuart–Landau. Under measurement noise the ESN was the more reliable model for Stuart–Landau relaxation (median 0.56–0.66 against DMD's 0.8–1.0, truth 0.58) **[have]**.
- No passive-data model recovers coupling direction **[have]**.
- **A linear model's cycle amplitude is neutral [have: slip_rotation.jl header; long-horizon check].** A linear model sustains a cycle only by pinning its fundamental on the unit circle, so the part of a perturbation that lands on the fundamental never decays. Over 30 steps, DMD's off-cycle deviation from its own run plateaus at 28–49% of the initial deviation after about 8 steps on SLIP, and at 20–47% on coupled FHN; the truth and the ESN return to their cycles. Its higher harmonics also decay slowly (0.98, 0.98, 0.97 per step on SLIP), so its waveform drifts. This bounds the DMD result in Fig 5A, B to the first few cycles. **[recheck the coupled-FHN plateau with the curve-distance scoring: late in a decay the deviation approaches the nearest-sample floor]**
- The claim is about meeting the requirements together, with failures that are known and analysable, not about universal accuracy.

**Figure 4: Self-driving benchmark (R1, R2)** **[have: Fig2_self_driving.jl]**.
- (A) Phase portraits in the realistic regime (12 training cycles, 1% measurement noise): free runs from the end of the training window, every seed, degree or rank overlaid.
- (B) Outcome after 30 free-running cycles, pooled over systems and regimes: ok / decays with the right rhythm / collapse / blow-up / wrong rhythm.
- (C) Forecast error over the first held-out cycle, per system, with the mean-cycle reference forecast (zero skill). Consider plotting skill directly.
- Supplement: forecast baselines **[have: FigS1]**.

**Figure 5: One fit, no seed spread; and what each model's cycle has learned (R1, R3)** **[have: Fig6_linear_affordances.jl, fig6_relaxation_data.jl]**.
- (A, B) Off-cycle relaxation per start and per fit over the first 4 cycles, coupled FHN and SLIP (centre of mass): DMD (one fit) against ESN and neural-ODE seeds, truth per start (ordered by the truth's rate). Neural ODE from cached runs with a provisional reference (mock-up until refitted). Stuart–Landau dropped (too simple; DMD reproducibly wrong at 10 decays).
- (C) SLIP: multipliers of each model's own cycle, rotation against modulus per step: DMD's operator eigenvalues in the cycle frame (the one next to the 8th harmonic at the exact multiplier), the ESN's monodromy eigenvalues, the lattice fit; neural ODE pending a refit. Option: a resolvent-gain curve of DMD on the unit circle to show the harmonics; it cannot show the sideband, which sits 0.04 cycles per step from the 8th harmonic's peak.
- The SLIP bias correction moved to Fig 2B, C; FB/TLS stay in the text only.
- Optional later: a residual-verified (ResDMD) spectrum on gait or SLIP.

Open: whether to add an LSTM (the gait-signatures model class) **[new]**. A reviewer from the gait-signatures community will ask.

---

## 5. Linear models on real gait: what they show, and where the data stop

**¶1. The transient lives in small deviations around a dominant mean cycle.** Most of gait variance lies in the mean cycle. That part is the cycle's shape, not the dynamics around it: modelling effort aimed at reproducing it is aimed at the part of gait that says least about how it is controlled.
- 95–99% of the variance of a 15-s trial is the mean cycle (deviations: median 1.4% AB, 4.7% LF) **[have: Fig 6B]**.
- Whole-trajectory operator measures inherit this. Transient growth of delay-DMD at a ⅓-stride window is 71–75% (log scale) reproduced by the mean cycle alone, group ordering included, because harmonics of the cycle are non-orthogonal over a window shorter than the stride **[have: Fig 6C]**.

**¶2. Identification success is not evidence of dynamics.** RNN-based gait signatures distinguish individuals robustly across speeds. Because they are stride-averaged summaries of an expressive model, it is unclear how much they reflect the mean cycle and how much the dynamics of deviations.
- Winner et al. 2023, 2024 [cite✓]. Lab-internal; frame as complementary.
- Test done **[have: Fig S5]**: the mean cycle identifies people at least as well as the operator signature (75% shape, 69% range-of-motion shares, 69% operator map of the mean cycle alone, against 63% for the operator map of the recorded trials). Identification success is therefore not evidence that a model has captured dynamics. Deviations alone also identify people (52–62%), so individual structure in variability exists. The same test should be run on RNN gait signatures (Winner et al.).

**¶3. Linear models of deviations target the transient directly.** A linear operator fitted to delay-embedded kinematics describes how deviations evolve: impulse responses, transient growth, and cross-joint transfer. These are the finite-time response operators of §2 ¶6, measured on the deviations, not on the mean shape.

**¶4. What 15-s trials can and cannot identify.** The quantities that matter (transverse, short-timescale, closed-loop) are exactly the ones passive on-cycle data constrain least. Perturbation-rich experiments and models whose estimation error can be analysed are therefore both needed.
- Stride-level dynamics are not identifiable from 15-s trials **[have: iv_gait_stride.jl]**. Section deviations barely carry over between strides: pooled lag-1 correlations −0.33 to +0.16 (negative at the stride cut) and lag-2 −0.33 to −0.02, comparable to the −1/n bias of removing each trial's mean over 6–12 strides. The instrumental-variable estimate (Fig 2) needs that lag-2 signal and returns anything from 0.06 to >40; least squares gives 0.1–0.3 with an unknown noise bias. Separating noise bias from stride-to-stride dynamics needs long recordings (hundreds of strides) or perturbations. Forecasting agrees: out of trial, no linear model (structured lattice, delay DMD on the deviations, companion operator) predicts the next stride's deviation better than the mean stride, even with the multiplier chosen at matched speed (Fig S6D). Independent agreement with far more data: phase-varying models (linear maps between phases, and a neural network) fitted to up to 240 strides per condition beat the mean cycle only up to 12.5–19% of a stride ahead (Rosenberg, Banjanin, Burden & Steele 2020 [cite?]).

**¶5. Verifiable spectra and imposed structure.**
- Residual DMD gives residuals that certify which eigenvalues and modes to trust (Colbrook & Townsend; Colbrook, Ayton & Szőke 2023 JFM [cite✓]).
- Multiresolution DMD separates timescales (Kutz, Fu & Brunton 2016 [cite?]).
- Physics-informed DMD imposes structure (Baddoo et al. 2023 [cite?]).
- DMD with control can use perturbation inputs (Proctor et al. 2016 [cite?]).
- **[new, optional]** ResDMD residuals on gait or SLIP eigenvalues.
- **Imposing the limit cycle's Koopman structure (exploratory) [have: floquet_lifted.jl; SLIP, 10 decays of 3 steps unless noted].** Theory: a stable limit cycle has Koopman eigenvalues on the lattice {ikω + mμ} (Mezić 2017; Mauroy & Mezić; Nakao & Mezić 2020 [cite?]), so a linear model of the state needs exact harmonics plus the same Floquet sideband next to every harmonic. Plain Hankel-DMD fits each eigenvalue separately and blends harmonic with sideband (0.038 cycles per step apart here). Tests:
  - Longer decays (30 steps, with or without stride kicks): harmonics lock, but no clean sideband, spurious near-neutral modes, plateau stays 0.17–0.60. Record length alone is not the fix.
  - ResDMD residuals on held-out decays trust the harmonics (including the neutral fundamental) and cannot separate the near-correct sideband (0.763 ∠ 13.9°, residual 0.050) from the blended one (0.827 ∠ 7.6°, 0.052); residual filtering leaves the plateau.
  - Phase-indexed monodromy (cycle subtracted, period-ahead linear map at every phase): unstable from 3-step decays; 0.77 ∠ 11° from 30-step kick-free decays.
  - **Lattice-constrained fit (one ω and one μ for all harmonics, variable projection): 0.771 ∠ 14.6° from the 3-step decays (truth 0.769 ∠ 13.6°)**, where plain DMD has no single answer. Its first-order forecast model (one shared mode shape) captures only 72% of the decays' amplitude.
  - Neural phase–isostable model (linear dynamics in (θ, ψ), neural decoder that vanishes on the cycle), μ frozen at the lattice estimate: returns to its cycle (plateau 0.03–0.08 against DMD's 0.28–0.49), the same answer for 3 seeds (±0.01), 4-step rates 0.71–0.80 (truth 0.68–0.90). With μ free it fits better but drifts to 0.84 ∠ 20°.
  - Stride kicks (process noise) make the phase diffuse and bias the lattice fit (0.95 ∠ 3°): for noise-driven gait the phase must be re-estimated per stride (Revzen & Guckenheimer's Phaser).
  - Reading for the paper: structure that theory says a limit cycle has can be imposed on a linear model, and then even short transients identify the multiplier. This keeps the linear model's strengths (§4 ¶2, ¶5) and is not a property of plain DMD.
  - **SLIP off-cycle metrics [have: Fig S6C; recompute the DMD, ESN and neural-ODE rows with Fig 5's corrected scoring].** 4-step relaxation-rate |error| (Fig 5's metric): lattice fit 0.24 (worst; one first-order mode), phase–isostable net 0.08 (μ from the lattice) / 0.14 (μ free), against DMD 0.035, ESN 0.04–0.17, neural ODE 0.046–0.055. Distance to the true cycle after 25 steps (fraction of initial): phase–isostable net 0.10–0.17, lattice 0.31, against DMD 0.52, ESN 0.36–0.53, neural ODE 0.43–0.71. So the structure buys the multiplier and the long-horizon return, not short-horizon accuracy.
  - **On noise-driven, stride-normalised data [have: gait_lattice.jl, Fig S6D–F]** (one Floquet mode; ψ fitted to half a stride, the next 2.5 strides forecast; μ chosen on the person's other trials, never on the scored strides). Kicked SLIP walks of 15 steps: R² 0.53 one stride ahead (deviation DMD 0.25, companion DMD −0.12), a clear error minimum at ρ 0.70 (truth 0.769), rotation unidentified. Human gait: no skill. R² one stride ahead AB 0.01, HF 0.03, LF −0.01 (positive in 35–58% of trials); a delay DMD on the deviations −0.12 to −0.34; the companion DMD −1.5 to −1.9. Forecast error against the assumed multiplier is flat at the mean stride's level, so per-person multipliers (AB 0.32, HF 0.45, LF 0.45; AB 0.43 when every trial is cut to 10 strides) are not identified. A person's trials are at different speeds (AB span about 2×), but speed is not what hides the multiplier: μ from the person's nearest-speed trial, or from the 10 nearest-speed trials of other people, also gives R² ≤ 0.05, and a trial's own best modulus does not track its speed in AB (within-person rank correlation 0.07; −0.49 in stroke, 9 and 3 people) **[have: gait_lattice_speed.jl]**. Choosing μ on the scored strides, as the first version of the script did, gives R² 0.07–0.16. About half of that (0.04–0.08) is selection: stride-shuffled copies of each trial give it too, while kicked SLIP walks drop from 0.57 to 0.05 **[have: gait_lattice_null.jl, Fig S6F]**. The rest is a small stride-order signal, clear only in HF (+0.06, people-bootstrap 95% 0.03–0.07; AB +0.01, LF +0.03 with a wide interval) and concentrated in trials whose own best modulus is near-neutral (≥ 0.8), which looks like persistence or drift; no rule that leaves out the scored strides recovers it. A cautionary example for any per-trial model selection. Reading: imposing the lattice helps where the data contain transients (SLIP decays); it cannot create stride-to-stride signal that passive 15-s trials lack (¶4).
  - **One family of limit-cycle models, truncated differently [text only].** Near a stable cycle there are coordinates in which the dynamics are exactly linear: an asymptotic phase θ that advances uniformly, and isostable amplitudes ψⱼ that each shrink (and turn) by their Floquet multiplier every cycle (Mauroy, Mezić & Moehlis 2013 [cite?]; Wilson & Moehlis 2016 PRE [cite?]). Everything nonlinear sits in the map back to the measurements, x = W(θ, ψ). The models in use differ in which part of W they keep:
    - Section return map: W read at one phase, once per stride. Linear, it gives the classical Floquet multipliers (Hurmuzlu & Basdogan 1994 [cite?]; Dingwell & Kang 2007 [cite✓]); as a sparse polynomial it is SINDy on the Poincaré map, "nonlinear Floquet theory" (Bramburger & Kutz 2020 Physica D [cite?]).
    - Phase-varying maps: a linear (or neural) map from phase φ to φ + Δ, fitted separately for every phase and horizon, with no multiplier shared between them (Revzen & Guckenheimer 2012 [cite✓]; Rosenberg, Banjanin, Burden & Steele 2020 J R Soc Interface [cite?]).
    - Lattice-constrained model (above): linear in ψ, every phase at once, one multiplier for all of them. Theory requires the sharing (the multipliers are the same at every section), and the sharing is what lets 3-step decays identify μ.
    - Higher-order isostable expansion, or the spectral submanifold of the cycle: W polynomial in ψ (the lattice's next rungs, ikω + 2μ and ikω + 2σ) plus amplitude-dependent decay and frequency (Wilson 2020 PRE [cite?]; Haller & Ponsioen 2016 [cite✓]). This is what the lattice lacks on SLIP's large early decays (4-step error 0.24, against 0.08 for a neural W).
    - Phase–isostable networks and phase autoencoders: neural W with linear latent dynamics (Yawata et al. 2024 Chaos [cite?]; the phase–isostable net above).
    - SSMLearn as run in Fig S1 (Cenedese et al. 2022 [cite✓]): a 2-D manifold that contains the cycle leaves one real direction off it, so it cannot represent a rotating multiplier such as SLIP's 0.769 ∠ 13.6°; and the cycle is learned as an attractor of the fitted map, not imposed.
  - **Options and limits with small data, which is typical of clinical gait [text only].** Clinical recordings are short: here 15-s treadmill trials of 3–19 strides (median 11), half of them for fitting; clinical gait analysis typically has only a few cycles (Rosenberg et al. 2020).
    - Return maps get one pair per stride, 6–12 per trial. A linear 12-D section map has 144 coefficients and a quadratic SINDy library 1,092. The linear map is already unidentifiable here (¶4), and sparsity cannot rescue a map whose linear part is not identified. Pooling trials for more pairs assumes one map across speeds and people.
    - The lattice is the most data-efficient option (every phase, two spectral numbers) and still finds nothing that transfers between trials (Fig S6D–F). Higher-order and neural W need large excursions to constrain their curvature, and passive walking does not produce them.
    - Phase-varying models fitted to up to 240 strides per condition beat the mean cycle only up to 12.5–19% of a stride ahead (Rosenberg et al. 2020). Hundreds of strides or perturbations are the entry ticket, not a refinement.
    - What small data can support: impose what theory fixes (exact harmonics, one shared multiplier), estimate few numbers, and report identifiability (error against the assumed multiplier, Fig S6E) and a shuffle null (Fig S6F) instead of a point multiplier.
  - **Two caveats for every member of the family [text only].**
    - The exact linear form lives in coordinates we do not measure. The phase and isostables are Koopman eigenfunctions of the whole neuromechanical system (muscle, neural and contact states included), which is infinite-dimensional in general. We record six joint angles. The map from the true (θ, ψ) to them is unknown, can mix several isostables into the same channels, need not be invertible, and comes with measurement noise. A one-mode model in joint angles is a projection of the true form, and its multiplier is at best the slowest one visible in those channels. Delay embedding can recover more in principle (Takens 1981 [cite✓]), but each delay dimension costs data that short trials do not have.
    - Finding the cycle and the phase is a problem of its own. Every member takes a cycle a(θ) and a phase θ(t) as given; from data both are estimates.
      - The mean cycle comes from the same few strides, and removing it biases stride-to-stride correlations by about −1/n (¶4).
      - The phase comes from a Hilbert transform or Phaser on noisy, non-sinusoidal kinematics (Revzen & Guckenheimer 2008 [cite✓]). Re-anchoring and resampling every stride, as here, removes phase diffusion, which is itself part of the dynamics. Leaving the diffusion in biases a fixed-frequency fit (0.95 ∠ 3° on kicked SLIP, above).
      - Off the cycle, any measured phase differs from the asymptotic phase by a term of first order in the deviation, the same order as the signal, so phase error and transverse dynamics are confounded.
      - Models that learn the cycle instead (DMD, ESN, neural ODEs, SSMLearn) can get it wrong outright: wrong period, collapse to a fixed point (§4 ¶2), detuned harmonics (Fig S6B).

**Figure 6: Gait recordings are mostly the mean cycle; the small deviations still change what a model reports** **[have: Fig5_gait_variability.jl, fig5_gait_data.jl; companion paper's 299 trials, 42 people]**.
- (A) One able-bodied trial: strides, mean cycle, deviations.
- (B) Deviation share of variance against strides per trial: median 1.4% (AB), 1.6% (HF), 4.7% (LF); 3–19 strides per 15-s trial (median 11). The slowest trials (24–43 cm/s, 4–9 strides) reach 15–45%.
- (C) The companion paper's transient growth (reimplemented, r = 1.000 with its per-trial values) against delay window: recorded trials vs the trial's own mean cycle repeated with no deviations. A periodic signal has nothing to amplify, yet over a window shorter than the stride its harmonics are non-orthogonal, so TG > 1: at the companion's ⅓-stride window the mean cycle alone gives TG 1.57× / 1.95× / 2.65× (AB/HF/LF), and 69% of the LF–AB difference. At a full-stride window it gives exactly TG = 1.
- (D) What the deviations add (per-trial Δ log₁₀ TG, recorded − mean cycle only): medians 0.03–0.10 (AB), 0.06–0.10 (HF), 0.09–0.20 (LF) across windows (×1.1–1.6), upper quartiles to 0.3. Deviations that are 1–5% of the variance raise TG by 10–60%, graded by group; at a full-stride window this is the whole effect (0.05 / 0.10 / 0.20).
- Supplement (FigS5_identification.jl): identification (companion split-half protocol, chance 2.4%): mean-cycle shape 75%, range-of-motion shares 69%, operator map of the recorded trials 63% (companion: 64.5%), operator map of the mean cycle alone 69%, deviation SD profile 62%, deviation cross-joint correlation 52%. Use in ¶2 as a sentence: identification success is not evidence that a model captured dynamics.
- **For the companion paper (flagged to Michael):** at its ⅓-stride window most of its TG and of its group differences is a mean-cycle/window property; the deviation part survives and is graded, and a full-stride window isolates it. Its 64.5% identification is matched by the mean cycle alone.

**Figure S6: Imposing the limit cycle's structure on a linear model** **[have: FigS6_lattice.jl]**.
- (A) The lattice model on one SLIP decay; (B) Koopman spectrum in the cycle frame, lattice against Hankel-DMD; (C) SLIP off-cycle metrics for all models; (D) gait and kicked-walk forecast skill, out of trial; (E) forecast error against the assumed multiplier (identifiability); (F) skill with μ chosen out of trial, on the scored strides, and on stride-shuffled copies (half of the on-test skill is selection; the rest does not transfer).

---

## 6. Discussion and recommendations

**¶1. Recommendation: judge models by how they predict, the way they will be used.** Score autonomous rollouts on held-out data as skill against reference forecasts (mean cycle, last stride), on the full signal and on the deviations, on and off the cycle, and select hyperparameters by rollout validation (preferably several rollouts from different starts), not by training or one-step loss (Fig S7). Then check long-run behaviour, seed spread, measurement noise and data size (§3 ¶1).

**¶2. Recommendation: collect transients.** Design experiments that excite transverse dynamics (perturbations, speed changes, split-belt). Without them, no model can identify stability. With clinical-length recordings (a few strides per condition), report identifiability and a null, not a multiplier (§5 ¶5).

**¶3. Recommendation: report finite-time response, not only asymptotic multipliers.** Compute it under uncertainty (bootstrap).

**¶4. Recommendation: prefer models whose errors can be analysed.** Use expressive models as forecasters or embeddings, not as sources of mechanistic stability claims.

**¶5. Outlook.** Closed-loop identification with perturbations and controlled inputs; linking operator structure to neural control (companion non-normality paper).

## 7. Conclusion (one paragraph)

---

## Restructuring of 2026-11-03

The paper is now organised as target → requirements → candidates → gait, not as "linear models beat expressive ones, and allow analyses". Script and figure filenames keep the old figure numbers, and script headers that cite OUTLINE.md sections use the old numbers below.

| Old | New |
|---|---|
| §2 Learnability | §3 ¶2–¶7 (now the reasons the requirements are hard on gait) |
| §3 Benchmark (¶1–¶4, ¶6) | §4 ¶1, ¶3, ¶8 |
| §3 ¶5 Template | §3 ¶1, R1 (the mean cycle as the zero-skill reference forecast) |
| §4 ¶1, ¶2, ¶4, ¶5, ¶6 Floquet | §2 ¶1, ¶5, ¶4, ¶2, ¶6 |
| §4 ¶3 Models that forecast well get multipliers wrong | §4 ¶3 |
| §5 ¶1, ¶3, ¶4, ¶5 | §5 ¶1, ¶2, ¶3, ¶4 |
| §5 ¶2 Variability carries the control signal | §2 ¶3 |
| §6 ¶1 Stability by construction | §4 ¶3 |
| §6 ¶2 One convex answer | §4 ¶2 (off-cycle results), ¶4 (one answer) |
| §6 ¶3 Cheap enough to validate | §4 ¶7 |
| §6 ¶4–¶5 Bias analysable; nonlinear models biased | §4 ¶5 |
| §6 ¶6 Verifiable spectra, lattice | §5 ¶5 |
| §6 ¶7 Limits of linear models | §4 ¶10 |
| §7 Discussion, §8 Conclusion | §6, §7 |

| Figure | Old number | Script |
|---|---|---|
| 1 Eigenvalues are not the whole story | 4 | Fig4_nonnormality.jl |
| 2 Data-driven Floquet maps (SLIP, IV) | 3 | Fig3_floquet_maps.jl |
| 3 Learnability | 1 | Fig1_learnability.jl |
| 4 Self-driving benchmark | 2 | Fig2_self_driving.jl |
| 5 One convex answer | 6 | Fig6_linear_affordances.jl |
| 6 Gait is mostly the mean cycle | 5 | Fig5_gait_variability.jl |

Also in this pass: Figures 1 and 2 were marked [new] but exist; their panel lists now follow the scripts. The identification test is Fig S5 (was cited as Fig 5D). New: §3 ¶1 (requirements R1–R5), §2 ¶3's first two sentences, §4 ¶1's framing, ¶6, ¶9 (scorecard). "Flexible" is now "expressive" throughout.

Prediction pass (same day): prediction is now the first requirement (R1, predictive skill in autonomous rollouts on held-out data, scored against the mean cycle and the last stride as zero-skill references); long-run fidelity is R2, and the old R2–R5 are R3–R6. §4 ¶2 is new (forecast skill from the Fig 4C caches, Fig S1, Fig 5, Fig S6D); validation by rollout rather than training loss is to-do 7.

## Changes from the Sep-10 draft (index.qmd)

- Drop the Stuart–Landau non-normality demonstration. The Figure 1 non-normality was an artefact (62 modes fitted to a rank-2 signal), and the SL multiplier of 0.533 was wrong.
- Re-check the claim "reservoir computing: complete failure on real gait". Tuned ESNs are much better than the untuned ones; restate against the tuned results.
- Remove the statement that linear models cannot represent radial relaxation (retracted; they can, given transient data).
- Move the piecewise-linear demonstration from Stuart–Landau/coupled FHN to SLIP/Van der Pol.
- Recast the "non-normality interpretation" section as one part of Section 2 plus a pointer to the companion paper.

## Analysis to-do list (priority order)

1. **Fig 1/2 systems.** SLIP and Van der Pol: Floquet maps, piecewise maps, monodromy transient growth. **[done: Fig3_floquet_maps.jl, Fig4_nonnormality.jl]**; replace the coupled-FHN numbers in §2 ¶5 with SLIP ones.
2. **Fig 5 reproducibility.** Coupled FHN and SLIP. **[done]**
3. **Matched-data off-cycle comparison** (ESN and neural ODE at 100 decays), Fig 3D. **[new; neural ODE takes hours]** Fills R5 for the expressive models.
4. **Bias-correction demo.** IV in Fig 2B, C. **[done]**
5. **Gait transient structure**, Fig 6. **[new; coordinate with companion paper]**
6. **Scorecard, Table 1** (§4 ¶9). **[new; from existing results]**
7. **Validation, not training loss.**
   - Fig 3B and Fig S2E–F against held-out rollout error **[done]**.
   - One-step against rollout validation on the DMD and ESN tuning grids **[done: validation_criteria.jl, Fig S7]**. The tuning scripts themselves still choose by one held-out rollout; switching them to the multistart rollout would change the ESN's chosen settings in Fig 4 (decide).
   - Gait neural ODEs: save the weights in gait_overparam.jl and refit, to add one-step validation error, and redo trials 36 and 90 with held-out forecasts **[new; hours]**.
   - Held-out trials as validation data are confounded by speed (each of a person's trials is at a different speed); held-out time within a trial is cleaner.
8. **Optional:** LSTM in the benchmark; neural-ODE pitfalls test; gait-signature residual-shuffle test; ResDMD on gait.
9. **References:** verify every [cite?]; build references.bib.

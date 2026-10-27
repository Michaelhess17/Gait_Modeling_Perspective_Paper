# Outline: revised perspective draft

Working title options

- *Learning walking from data: why flexible dynamical models fail on gait, and what linear operator models offer*
- *Beyond the mean gait: transient structure, learnability, and linear models of locomotor dynamics*

Format: perspective (e.g., J. R. Soc. Interface / PLOS Comput Biol perspective), ~6 main figures plus a supplement.

Marks used below:

- **[have]**: the result exists in this repo (the source file is named).
- **[new]**: needs new analysis.
- **[pending]**: running now.
- **[cite✓]**: a reference checked during drafting.
- **[cite?]**: a reference from memory that needs checking before use.

---

## Abstract (sketch, 5 sentences)

1. Data-driven dynamical models (neural ODEs, reservoir computers, sparse regression, spectral-submanifold reductions) succeed on simulated systems and fluid flows, and are increasingly applied to human walking.
2. We show that in free-running (self-driving) mode these models frequently fail on realistic biological oscillators and on gait, by blowing up, settling on the wrong attractor, or collapsing. Linear Hankel-DMD stays stable and qualitatively correct.
3. We argue the failures reflect learnability rather than implementation. Walking is a closed-loop, noisy, hybrid system observed briefly and near its limit cycle, so the data constrain the mean cycle well but the transverse dynamics poorly.
4. The usual target, asymptotic Floquet multipliers, may be the wrong quantity: it is data-hungry, noise-sensitive, and incomplete for the inherently non-normal return maps of oscillators. We argue for transient, stride-scale structure and for the origins of variability.
5. Linear operator models give us what flexible models do not: stability by construction, a convex and reproducible fit, analyzable and correctable estimator bias, and verifiable spectra and modes.

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

**¶4. State of the field: data-driven models.** A third, newer tradition fits dynamical models to gait data. The methods range from linear step-to-step maps to deep sequence models, and each trades interpretability for flexibility differently.
- Step-to-step linear maps that infer control from variability: Wang & Srinivasan 2014 [cite?]; Seethapathi & Srinivasan 2019 eLife [cite✓]. Goal-equivalent manifolds: Dingwell & Cusumano [cite?].
- Phase-based models: Revzen & Guckenheimer 2008/2012 [cite?]; phase-varying models of exoskeleton response, Rosenberg et al. 2020 [cite?]; Hybrid-SINDy for centre-of-mass dynamics, Rosenberg, Proctor & Steele 2024 Sci Rep [cite✓].
- Neural models: RNN gait signatures, Winner et al. 2023 PLOS Comput Biol [cite✓], extended 2024 Sci Rep [cite✓]; modulated neural ODEs evaluated on human walking trajectories (MoNODE, NeurIPS 2023) [cite✓].
- Koopman/DMD in gait: Fujii et al. 2019 Sci Rep (DMD for coordinative structures) [cite✓]; Koopman model of whole-body angular momentum with periodic regularization (NSF PAR 10613388) [cite✓, get the full reference].

**¶5. The gap.** Despite this activity, there is little systematic evidence on whether flexible data-driven models recover the dynamics of walking, as opposed to reproducing its average shape. Methods that work convincingly on simulated benchmarks and fluid flows are often assumed to transfer.

**¶6. This perspective.** We (i) test state-of-the-art models in the regime that matters, free-running prediction, on realistic oscillators and gait; (ii) explain their failures as a learnability problem specific to closed-loop locomotion; (iii) argue that asymptotic Floquet structure is the wrong target and transient, variability-generating structure is the right one; and (iv) show what linear operator models make possible that flexible models do not.

---

## 2. Why gait is not flow past a cylinder: learnability

**¶1. Where data-driven dynamics succeed.** Neural ODEs, SINDy, reservoir computers and DMD have their clearest successes on systems with fixed governing equations, high signal-to-noise ratio, abundant snapshots, and dynamics that are observed through their transients.
- Cylinder wake (Bagheri 2013 [cite?]), Lorenz/KS with reservoirs (Pathak et al. 2018 [cite?]), SINDy (Brunton et al. 2016 [cite?]), neural ODEs (Chen et al. 2018 [cite?]).
- In fluids, start-up transients and parameter sweeps supply off-attractor data cheaply.

**¶2. Walking is a closed loop.** In walking, mechanics and neural control form one feedback loop. Passive recordings of the loop cannot separate plant from controller, a classic result in closed-loop system identification.
- Forssell & Ljung 1999 [cite?].
- Standing balance needs deliberate perturbations to separate plant and feedback (van der Kooij, Pasma, Schouten and colleagues; closed-loop system identification) [cite✓].
- Our coupling test: no model, linear or nonlinear, recovers coupling direction from passive data **[have: coupling_benchmark.jl, cache/coupling]**. One sentence plus a supplement pointer.

**¶3. Feedback hides the transverse dynamics.** Because the controller suppresses deviations, walking data stay close to the limit cycle. The transverse directions that encode stability and control are excited only weakly, by intrinsic noise.
- The mean cycle explains about 94% of gait variance **[have]**.

**¶4. Short, hybrid, partially observed, nonstationary.** Gait datasets add further obstacles: tens of strides per trial, a few kinematic channels standing in for a high-dimensional neuromechanical state, discrete contact events, and slow drift in speed and posture.

**¶5. Consequence: on-cycle data identify the cycle, not the dynamics around it.** Many models fit on-cycle data equally well yet disagree about everything off the cycle, so training loss cannot select among them.
- Gait neural-ODE fits reach equal loss with learned periods from 0.7× to 21× the stride and different multipliers **[have: Fig_node_gait, gait_overparam.jl]**.
- Floquet multipliers read from models trained on on-cycle data are unconstrained: Stuart–Landau neural ODE ≈ 1.0 and DMD ≈ 0.99 against a truth of 0.60 **[have: floquet_tuned.jl]**.

**¶6. Transient-rich data make the problem learnable, given enough of it.** When training data contain decays from perturbed states, transverse relaxation becomes identifiable. Even then, recovering it takes many transients and a delay window long enough to span the decay. Nothing is free.
- Hankel-DMD relaxation rate converges to the truth as decays increase (coupled FHN from about 30 decays; Stuart–Landau needs at least 1.5-cycle windows and about 300 decays) **[have: Fig_dmd_datasize, dmd_delay_sweep.jl]**.
- Experimental implication: perturbation paradigms (treadmill perturbations, split-belt, tripping) are the gait analogue of start-up transients.

**Figure 1: Learnability.**
- (A) Schematic: on-cycle data occupy a thin tube; perturbation data fill the transverse directions; closed-loop diagram.
- (B) Equal-loss neural-ODE fits to gait diverge **[have]**.
- (C) DMD relaxation rate against amount of transient data **[have]**.
- (D, optional) Off-cycle relaxation of all models at matched data **[pending: ESN/NODE retune at 100 decays]**.

---

## 3. Main result: free-running models fail where linear Hankel-DMD holds

**¶1. Why free running is the right test.** A dynamical model is only useful as a model of dynamics if, released from data, it generates the system's behaviour. One-step or teacher-forced accuracy hides exactly the failures that matter.

**¶2. Benchmark design.**
- Systems: coupled FitzHugh–Nagumo, coupled Van der Pol, Stuart–Landau, SLIP and human gait.
- Regimes: varied data length and noise level.
- Models: neural ODE, ESN, LSTM?, spectral-submanifold model (SSMLearn) and Hankel-DMD, each tuned on training data only with wide hyperparameter searches.
- **[have: self_driving_benchmark.jl, dmd_tuning.jl, esn_tuning.jl]**

**¶3. Results.** Hankel-DMD sustains the correct cycle in 92% of system–regime cells and never blows up. The nonlinear models succeed in 42–62% and blow up or settle on the wrong attractor in 12–19% **[have: Fig_self_driving]**.

**¶4. How each model fails.** The failure modes differ in kind. Nonlinear models fail by blowing up, by settling on a wrong period, or by collapsing to a fixed point. DMD fails by decaying slowly, which is benign and predictable because the |λ| ≤ 1 constraint guarantees it.

**¶5. The template caveat.** Better forecasting is not the argument: a phase-indexed mean-cycle template forecasts gait better than every model, which shows that the mean is easy and that forecast error measures the wrong thing.
- Template 0.17/0.27/0.39 against the spectral-submanifold model 0.28/0.38/0.62 **[have: Fig_forecast_baselines]**.

**¶6. Implementation is not the explanation.** These failures persist with careful training: multiple shooting, adjoint gradients, RK4 integration, and train-only selection over wide searches. Common pitfalls make them worse (Box 1).
- **Box 1 / supplement: neural-ODE pitfalls.** Forward Euler at the sample step spirals outward on oscillations; an output nonlinearity biases the vector field; very short shooting windows; latent encoder/decoder non-identifiability; summed weight decay; selection on minibatch loss. **[new, optional: a controlled degradation test on Stuart–Landau, about 1 h]**

**Figure 2: Self-driving benchmark** **[have: Fig_self_driving]**. Supplement: forecast baselines **[have]**.

Open: whether to add an LSTM (the gait-signatures model class) **[new]**. A reviewer from the gait-signatures community will ask.

---

## 4. Floquet multipliers may be the wrong target

**¶1. Why Floquet multipliers became the standard.** Floquet multipliers answer a precise question: how fast, asymptotically, a perturbation shrinks per cycle. That made them the natural stability measure for gait.
- Hurmuzlu & Basdogan 1994 [cite?]; Dingwell & Kang 2007 [cite✓].

**¶2. They are data-hungry and noise-fragile.** Estimating multipliers from return maps needs many cycles and low noise. Piecewise-linear phase-sector maps compound the problem, because noise attenuation multiplies across sectors.
- Revzen & Guckenheimer [cite?]; Riva et al. 2013 [cite✓].
- Our maps: phase map 0.59 [0.50–0.68] at 30 cycles against a truth of 0.69; the 8-sector piecewise map gives 0.41 and collapses to about 0 at noise ≥ 1× **[have: Fig_floquet_maps, on coupled FHN]**.
- **[new]** Redo on SLIP (hybrid; SLIP.jl `paper_stability.jl` has self-stable walking with known multipliers) and Van der Pol (relaxation oscillator), replacing Stuart–Landau and coupled FHN as the showcase systems.

**¶3. Models that forecast well can still get the multipliers wrong.** Read out of fitted models, Floquet multipliers vary widely across model classes, seeds and hyperparameters, even among models that forecast similarly.
- Neural ODE on coupled FHN 0.01 against a truth of 0.69; ESN 0.64 [0.38–0.86]; DMD 0.71 [0.66–0.79] **[have: floquet_tuned.jl]**.
- Honest caveat: on noisy Stuart–Landau the ESN is closest (0.57 against 0.60).

**¶4. Return maps of oscillators are inherently non-normal.** Even for a textbook Van der Pol oscillator, the monodromy matrix and stride-to-stride return map are non-normal: their eigenvectors are far from orthogonal, and within-cycle variational dynamics amplify perturbations transiently before they decay. Eigenvalues are not the full story.
- Trefethen & Embree 2005; Trefethen et al. 1993 "Hydrodynamic stability without eigenvalues" [cite?].
- Bridge to the companion paper; keep this brief.

**¶5. Walking never reaches the asymptotic regime.** Gait is perturbed every stride by intrinsic noise and by the environment, so what governs behaviour is the response over the next one to three strides, the transient. Asymptotic multipliers can miss it entirely.

**¶6. A better target: finite-time, stride-scale response.** We propose characterizing gait by its finite-time response operators: how deviations at one stride propagate, amplify and redistribute across joints over the next few strides. These are quantities that linear models expose directly.

**Figure 3: Data-driven Floquet maps on SLIP / Van der Pol** **[new; reuse the floquet_maps_benchmark.jl pipeline]**.
- (A) Return map of SLIP apex state (or Van der Pol section) at realistic and large cycle counts.
- (B) Multiplier estimate against record length.
- (C) Against noise.
- (D) Piecewise-linear sector maps: bias grows with the number of sectors.

**Figure 4: Eigenvalues are not the whole story** **[new]**.
- (A) Van der Pol (and SLIP) monodromy: eigenvalue against the norm of the n-stride map over strides, showing transient amplification despite |μ| < 1.
- (B) Two systems with matched multipliers but different transient responses.
- (C) Within-cycle growth of perturbations (variational dynamics), phase-resolved.

---

## 5. From the mean gait to the origins of variability

**¶1. The mean cycle is easy and uninformative about control.** Most of gait variance lies in the mean cycle, and a template captures it better than any dynamical model. Modelling effort aimed at reproducing the mean is aimed at the part of gait that says least about how it is controlled.
- 98–99% of the variance of a 15-s trial is the mean cycle (deviations: median 1.4% AB, 4.7% LF) **[have: Fig 5B]**.
- Whole-trajectory operator measures inherit this. Transient growth of delay-DMD at a ⅓-stride window is 71–75% (log scale) reproduced by the mean cycle alone, group ordering included, because harmonics of the cycle are non-orthogonal over a window shorter than the stride **[have: Fig 5C]**.

**¶2. Variability carries the control signal.** Stride-to-stride deviations, and how they are corrected, reveal the control strategy.
- Seethapathi & Srinivasan 2019 [cite✓]; Wang & Srinivasan 2014 [cite?]; goal-equivalent manifolds, Dingwell & Cusumano [cite?]; uncontrolled manifold, Scholz & Schöner 1999 [cite?]; minimal intervention, Todorov & Jordan 2002 [cite?].

**¶3. Gait signatures and what they capture.** RNN-based gait signatures distinguish individuals robustly across speeds. Because they are stride-averaged summaries of a flexible model, it is unclear how much they reflect the mean cycle and how much the dynamics of deviations.
- Winner et al. 2023, 2024 [cite✓]. Lab-internal; frame as complementary.
- Test done **[have: Fig 5D]**: the mean cycle identifies people at least as well as the operator signature (75% shape, 69% range-of-motion shares, 69% operator map of the mean cycle alone, against 63% for the operator map of the recorded trials). Identification success is therefore not evidence that a model has captured dynamics. Deviations alone also identify people (52–62%), so individual structure in variability exists. The same test should be run on RNN gait signatures (Winner et al.).

**¶4. Linear models of deviations target variability directly.** A linear operator fitted to delay-embedded kinematics describes how deviations evolve: impulse responses, transient growth, and cross-joint transfer. These are variability-generating structure, not mean shape.

**¶5. Learnability, again.** The quantities that matter (transverse, short-timescale, closed-loop) are exactly the ones passive on-cycle data constrain least. Perturbation-rich experiments and models whose estimation error can be analysed are therefore both needed.
- Stride-level dynamics are not identifiable from 15-s trials **[have: iv_gait_stride.jl]**. Section deviations barely carry over between strides: pooled lag-1 correlations −0.33 to +0.16 (negative at the stride cut) and lag-2 −0.33 to −0.02, comparable to the −1/n bias of removing each trial's mean over 6–12 strides. The instrumental-variable estimate (Fig 6) needs that lag-2 signal and returns anything from 0.06 to >40; least squares gives 0.1–0.3 with an unknown noise bias. Separating noise bias from stride-to-stride dynamics needs long recordings (hundreds of strides) or perturbations.

**Figure 5: Gait recordings are mostly the mean cycle; the small deviations still change what a model reports** **[have: Fig5_gait_variability.jl, fig5_gait_data.jl; companion paper's 299 trials, 42 people]**.
- (A) One able-bodied trial: strides, mean cycle, deviations.
- (B) Deviation share of variance against strides per trial: median 1.4% (AB), 1.6% (HF), 4.7% (LF); 3–19 strides per 15-s trial (median 11). The slowest trials (24–43 cm/s, 4–9 strides) reach 15–45%.
- (C) The companion paper's transient growth (reimplemented, r = 1.000 with its per-trial values) against delay window: recorded trials vs the trial's own mean cycle repeated with no deviations. A periodic signal has nothing to amplify, yet over a window shorter than the stride its harmonics are non-orthogonal, so TG > 1: at the companion's ⅓-stride window the mean cycle alone gives TG 1.57× / 1.95× / 2.65× (AB/HF/LF), and 69% of the LF–AB difference. At a full-stride window it gives exactly TG = 1.
- (D) What the deviations add (per-trial Δ log₁₀ TG, recorded − mean cycle only): medians 0.03–0.10 (AB), 0.06–0.10 (HF), 0.09–0.20 (LF) across windows (×1.1–1.6), upper quartiles to 0.3. Deviations that are 1–5% of the variance raise TG by 10–60%, graded by group; at a full-stride window this is the whole effect (0.05 / 0.10 / 0.20).
- Supplement (FigS5_identification.jl): identification (companion split-half protocol, chance 2.4%): mean-cycle shape 75%, range-of-motion shares 69%, operator map of the recorded trials 63% (companion: 64.5%), operator map of the mean cycle alone 69%, deviation SD profile 62%, deviation cross-joint correlation 52%. Use in §5 ¶3 as a sentence: identification success is not evidence that a model captured dynamics.
- **For the companion paper (flagged to Michael):** at its ⅓-stride window most of its TG and of its group differences is a mean-cycle/window property; the deviation part survives and is graded, and a full-stride window isolates it. Its 64.5% identification is matched by the mean cycle alone.

---

## 6. What linear operator models make possible

**¶1. Stability by construction.** Constraining eigenvalues guarantees bounded free runs, so the worst-case linear model is a slowly decaying one, not an exploding or wrong one **[have: self-driving fates]**.

**¶2. One convex answer.** The DMD fit is a least-squares problem with a unique solution for each hyperparameter setting: no seeds, no local minima, and results that vary smoothly with rank and delay. Flexible models give different answers at equal loss.
- Off-cycle relaxation of decay-trained models, same 10 decays for every model **[have: Fig 6A]**. Coupled FHN (truth 0.74; 0.70–0.84 across start phases): DMD 0.80 (0.76–0.82 across start phases); ESN 0.88–0.97 over 4 seeds (0.58–1.15 across start phases); neural ODE 0.61–0.73 over 3 seeds (0.14–1.15). Stuart–Landau (truth 0.58): DMD 0.88, reproducibly wrong at this data size (it needs about 300 decays, Fig 1C); ESN 0.46–0.48; neural ODE 0.49–0.54.
- Honest reading: on these toy systems the seed spread is moderate. The large reproducibility failure is on gait (Fig 1B). DMD's advantage is that a wrong answer is the same wrong answer every time, and converges visibly with data.

**¶3. Cheap enough to validate.** Fits take seconds, which makes cross-validation, bootstrap over trials and subjects, and wide hyperparameter sweeps routine. A single neural-ODE fit takes hours **[have: timing from runs]**.

**¶4. Estimator bias is analysable and correctable.** In linear models we know how estimation errors bias the result and can correct it. Least squares treats the predictors X as noise-free and the targets Y as noisy; with measurement noise this pulls eigenvalues inward. Forward–backward and total-least-squares DMD remove that bias, and optimized/bagged DMD adds uncertainty estimates.
- Dawson et al. 2016 [cite?]; Hemati et al. 2017 [cite?]; Askham & Kutz 2018 [cite?]; Sashidhar & Kutz 2022 [cite?].
- Honest framing from our tests: with rank-truncated Hankel matrices and decay data, plain least squares was already nearly unbiased up to 20% measurement noise on coupled FHN. TLS/FB did not improve and slightly slowed relaxation **[have: dmd_noise_variants.jl, noise_models.jl]**.
- **Demonstrated on the SLIP section map (Fig 6B–D) [have: fig6_bias_correction.jl].** Least squares attenuates: open loop 0.95 → 0.89 / 0.76 / 0.48 at noise 0.25 / 0.5 / 1× the stride-to-stride SD, and more strides do not help (0.87–0.89 from 30 to 2000 strides at 0.25×). Instrumental variables (previous stride as the instrument, no knowledge of the noise needed) recover the truth at every noise level: 0.945–0.948 open loop, and 0.86–0.87 against 0.872 with feedback.
- **The correction must match the noise model.** Forward–backward DMD returns |λ| = 1.000 exactly, at every noise level and even without noise: on a stationary, noise-driven process the backward fit has the same lagged statistics as the forward one, so FB reports neutral stability. TLS works only where its assumptions hold (open loop 0.91–0.96; feedback 0.90 at no noise, 0.61 at 1×). Unperturbed walking is a stationary, noise-driven process, which explains why TLS/FB did not help in our Hankel-DMD tests.

**¶5. Nonlinear models are biased by construction, and corrections are rare.** Neural ODEs, LSTMs and reservoirs trained by one-step or shooting loss inherit the same errors-in-variables bias, with no closed-form handle on it. Explicit corrections exist mostly for sparse regression.
- Simultaneous denoising for SINDy, Kaheman, Brunton & Kutz 2020 [cite✓]; neural ODEs with an implicit denoising network, Goyal & Benner 2023 [cite✓]; time-reversal-symmetry penalty, Huh et al. 2020 NeurIPS [cite✓] (closest analogue to forward–backward consistency).
- No total-least-squares or errors-in-variables neural ODE found (state as "to our knowledge").

**¶6. Verifiable spectra and multiscale extensions.**
- Residual DMD gives residuals that certify which eigenvalues and modes to trust (Colbrook & Townsend; Colbrook, Ayton & Szőke 2023 JFM [cite✓]).
- Multiresolution DMD separates timescales (Kutz, Fu & Brunton 2016 [cite?]).
- Physics-informed DMD imposes structure (Baddoo et al. 2023 [cite?]).
- DMD with control can use perturbation inputs (Proctor et al. 2016 [cite?]).
- **[new, optional]** ResDMD residuals on gait or SLIP eigenvalues.

**¶7. Limits of linear models.**
- Linearity is not a free lunch. DMD needs transient-rich data and long enough delay windows. Its settings matter on Stuart–Landau. Under measurement noise the ESN was the more reliable model for Stuart–Landau relaxation (median 0.56–0.66 against DMD's 0.8–1.0, truth 0.58) **[have]**.
- No passive-data model recovers coupling direction **[have]**.
- The claim is about structure and analysability, not universal accuracy.

**Figure 6: Linear-model affordances** **[have: Fig6_linear_affordances.jl]**.
- (A) Coupled FHN off-cycle relaxation per fit and per start phase: DMD (one fit) against ESN and neural-ODE seeds. Stuart–Landau dropped (too simple; DMD reproducibly wrong at 10 decays). A second, gait-like system (SLIP or Van der Pol) would need new ESN/NODE fits.
- (B, C) SLIP section-map ρ̂ against measurement noise: LS, FB, TLS, IV; open loop and with speed feedback.
- (D) Against record length at noise 0.25×: LS converges to the wrong value, IV to the truth.
- Optional later: a residual-verified (ResDMD) spectrum on gait or SLIP.

---

## 7. Discussion and recommendations

**¶1. Recommendation: evaluate models free-running.**

**¶2. Recommendation: collect transients.** Design experiments that excite transverse dynamics (perturbations, speed changes, split-belt). Without them, no model can identify stability.

**¶3. Recommendation: report finite-time response, not only asymptotic multipliers.** Compute it under uncertainty (bootstrap).

**¶4. Recommendation: prefer models whose errors can be analysed.** Use flexible models as forecasters or embeddings, not as sources of mechanistic stability claims.

**¶5. Outlook.** Closed-loop identification with perturbations and controlled inputs; linking operator structure to neural control (companion non-normality paper).

## 8. Conclusion (one paragraph)

---

## Changes from the Sep-10 draft (index.qmd)

- Drop the Stuart–Landau non-normality demonstration. The Figure 1 non-normality was an artefact (62 modes fitted to a rank-2 signal), and the SL multiplier of 0.533 was wrong.
- Re-check the claim "reservoir computing: complete failure on real gait". Tuned ESNs are much better than the untuned ones; restate against the tuned results.
- Remove the statement that linear models cannot represent radial relaxation (retracted; they can, given transient data).
- Move the piecewise-linear demonstration from Stuart–Landau/coupled FHN to SLIP/Van der Pol.
- Recast the "non-normality interpretation" section as one part of Section 4 plus a pointer to the companion paper.

## Analysis to-do list (priority order)

1. **Fig 3/4 systems.** SLIP (SLIP.jl stable walking, known multipliers) and Van der Pol: Floquet maps, piecewise maps, monodromy transient growth. **[new; about 1–2 days]**
2. **Fig 6A reproducibility.** From existing caches. **[new; about half a day]**
3. **Matched-data off-cycle comparison** (ESN and neural ODE at 100 decays), Fig 1D. **[new; neural ODE takes hours]**
4. **Bias-correction demo**, Fig 6B. **[new]**
5. **Gait transient structure**, Fig 5. **[new; coordinate with companion paper]**
6. **Optional:** LSTM in the benchmark; neural-ODE pitfalls test; gait-signature residual-shuffle test; ResDMD on gait.
7. **References:** verify every [cite?]; build references.bib.

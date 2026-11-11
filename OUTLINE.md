# Outline: modeling how gait responds and recovers

Revised 2026-11-04; reframed 2026-11-05 for a general audience interested in how gait dynamics are modeled, not a clinical one. Impairment is one application (§6), not the motivation.

Working title: *Beyond gait stability: modeling the dynamics of walking and recovery*

## The central argument

Understanding walking as a dynamical system requires more than knowing whether gait is asymptotically stable. We need to understand how a disturbance develops, how the body and its controller respond, which movements compensate, and how walking recovers. A Floquet multiplier compresses that behavior into a long-term rate: it can miss substantial short-term amplification and coordination changes, and it is difficult to estimate reliably from realistic recordings.

The alternative is a model of the evolving gait dynamics that can predict and help explain meaningful responses. Such a model must meet practical requirements: accurate recovery predictions, credible sustained behavior, reproducible estimates, interpretable structure, and feasible data and computational demands. Across these requirements, linear models in suitable coordinates offer the strongest practical starting point among the approaches examined here. Their value is the combination of those strengths, not a claim that they win every forecasting comparison.

**Narrative:** what a model of gait dynamics should capture → why an asymptotic stability number is inadequate → what a useful model must capture → which approaches meet those needs → how linear models support richer analysis of gait.

Data requirements support this argument. They should not displace the modeling question or become the organizing theme.

## Abstract

1. Data-driven dynamical models are increasingly fitted to walking, yet gait dynamics are still most often summarized by asymptotic stability. But walking is a rhythm that is disturbed and corrected at every stride; what characterizes it is how it maintains, adapts, and recovers, not only whether it converges to a periodic orbit.
2. Asymptotic stability overlooks important aspects of that behavior. Non-normal dynamics can amplify disturbances temporarily; active control and repeated disturbances shape the observed response. Estimation difficulties further limit the interpretation of a measured stability number.
3. We argue for models that describe the evolution of gait and predict responses to disturbances, with enough structure to study coordination and recovery.
4. We define the requirements for these models and compare representative approaches. Linear models offer a particularly useful combination of prediction, reproducibility, tractable estimation, and accessible dynamical analysis.
5. Realizing that promise requires appropriate excitation and validation. Reproducing the gait cycle alone does not demonstrate that a model captures how walking responds and recovers.

## 1. What do we need to understand about walking?

Open with the modeling question: walking is a rhythm produced by mechanics, neural control, and the environment, disturbed and corrected at every stride. What should a dynamical model of it capture, and how should such models be judged?

- Walking emerges from interacting body mechanics, neural control, and the environment. Changes in speed, task, terrain, or the body itself alter several parts of that interaction at once.
- The quantities of interest are the size and duration of an excursion, the joints or limbs involved, how coordination shifts, and the timing of recovery.
- Describing the usual gait pattern tells us what happened. A useful dynamical model should also predict what happens next and help organize an explanation of the response.
- Introduce stability measures as an influential approach, then establish why they cannot carry that full explanatory burden.

End with the thesis: understanding gait requires models of its evolving dynamics, judged against the needs of studying recovery; linear models provide a practical foundation.

## 2. Why asymptotic stability is not the right primary target

### A long-term return rate does not describe the recovery that matters

Explain Floquet stability briefly: whether sufficiently small deviations eventually shrink, and their asymptotic rates of change from cycle to cycle.

- A disturbance can initially grow before it decays. Non-normal interactions allow different transient responses even when asymptotic stability is the same.
- Direction and timing matter. The same eventual return rate can accompany different peak excursions, recovery times, and patterns of compensation.
- Walking never reaches the asymptotic regime. Disturbances (each step's ground collision, step-to-step foot-placement corrections, terrain, neural variability) arrive faster than responses decay: at multipliers of 0.3–0.8 per stride, a disturbance needs 2–10 strides to fall to a tenth. The observed deviation is a superposition of young responses, most less than a stride or two old, so stride-to-stride variability is a property of the short-time response.
- Observed behavior reflects the combined mechanics and controller; a multiplier does not separate their contributions.
- Persistent timing changes and altered coordination can matter even when walking remains bounded.

**Reader’s takeaway:** a gait can be asymptotically stable while exhibiting responses that matter for how walking behaves. The full response is the object of interest; its asymptotic rate is one summary.

**Figure 1: walking lives in the transient, and the transient is not the multiplier.** Fig1_transient.jl / fig-transient (data: fig1_transient_data.jl, ground truth only). (A) SLIP with speed feedback kicked at every step: the observed deviation as the sum of overlapping responses. (B) Within a stride: one coupled-FHN disturbance, typical and worst case, against the multiplier. (C) Stride to stride: worst-case n-stride growth against ρⁿ. (D) Share of the observed deviation from disturbances younger than n strides: coupled FHN, SLIP, and the human multiplier range. Merged 2026-11-05 with the former non-normality figure; its oscillator family (Van der Pol sweep, Stuart–Landau) and phase-shift panels are now FigS8_oscillators.jl / suppfig-oscillators.

*(Former figure: stability and recovery can tell different stories, Fig4_nonnormality.jl / fig-nonnormal. Merged into Figure 1; see above.)*

### Even the stability number is difficult to recover from gait data

Keep the practical challenge immediately after the conceptual critique. The target is incomplete, and its estimation is challenging.

- Short recordings provide few independent strides despite many within-stride samples.
- Measurement noise, state and event definitions, and approximation affect return-map estimates.
- Dividing the cycle into phase-dependent or piecewise models adds estimation and composition problems. More detail does not automatically produce a more reliable multiplier.
- Accurate waveform reconstruction does not establish correct off-cycle dynamics.
- Floquet multipliers have no central limit theorem, so larger data sizes do not fix the inherent finite-sample bias

**Figure: practical challenges in calculating Floquet stability.** Retain Fig3_floquet_maps.jl / fig-floquet-maps here in the main narrative, including its record-length, noise, and piecewise-model comparisons. Withdraw the earlier plan to relocate the whole figure to a late bias-correction section and automatically move the piecewise panels to the supplement.

Introduce the linear-map noise-bias correction briefly within this figure. Refer back to it later as a constructive benefit of linear estimation; it should not replace the figure’s role in establishing the practical Floquet problem.

## 3. What should a useful model of gait dynamics provide?

Bridge explicitly: replacing one stability estimate with a more complicated model only helps if the model answers questions the stability estimate cannot.

We want a model of the gait pattern **and the evolution of departures from it**, including disturbances, coordination, and recovery. “Holistic” means representing these interacting behaviors together, not claiming to reconstruct every physiological mechanism from kinematics.

### The capabilities that matter

1. **Forecast the response to disturbances.** From an observed state, predict how the deviation from the usual cycle evolves over the following strides (size, direction, timing, return) better than reference forecasts such as the mean cycle. This covers ordinary walking, whose deviations are themselves responses to ongoing disturbances, and larger disturbances of the kinds represented in training; no claim is made about conditions or kinds of walking absent from the training data. Assess timing errors separately from amplitude and coordination errors.
2. **Remain credible beyond a short fitted segment.** Produce plausible sustained walking and bounded trajectories over the intended observation window.
3. **Give reproducible answers.** Conclusions about a person or condition should not depend strongly on arbitrary fitting initializations.
4. **Work with realistic recordings.** Establish sensitivity to noise, record length, partial observation, and available excitation, and make uncertainty and data needs visible. Practicality belongs here (2026-11-11): a model that works on real data must be practical to fit and validate on it, without prohibitive computation or opaque tuning.
5. **Expose interpretable dynamical structure.** Help examine how disturbances spread between movements, where amplification occurs, and which responses dominate.
6. **Support comparison across people and conditions.** Fitted models should share a structure in which the same quantities can be computed and compared between participants, conditions, and tasks, so that differences reflect the dynamics rather than the fit. (Revised 2026-11-11: "practical to fit" moved to R4.)

### What data are needed to meet those requirements?

Present this as a practical requirement for learning recovery, not a separate opening argument.

- Repeated similar strides describe the usual gait pattern well, but may reveal little about substantial disturbances.
- Train on relevant excursions and test on independent perturbations within the stated domain, including different directions, phases, and magnitudes.
- Distinguish prediction from the moment of a disturbance from prediction after observing part of its recovery.
- Passive recordings can be informative, but their usefulness must be demonstrated for the response being claimed.

**Possible supporting figure: observing the gait cycle versus learning recovery.** The replacement learnability experiment is unfinished. Include it only if it makes this point clearly and economically. It supports the requirements; it is not a new organizing theme or a mandatory extra main-text figure.

**Figure (moved up 2026-11-11):** Fig5_gait_variability.jl / fig-gait-mean. Gait recordings are mostly the mean cycle, and whole-trajectory operator measures inherit it; this is the concrete challenge of gait recordings for dynamical modeling. §6 refers back to it.

## 4. How well do existing modeling approaches meet those needs?

Introduce Hankel-DMD, reservoir computing/ESNs, neural ODEs, and the other tested approaches through what they offer a gait researcher. Explain technical details where they account for strengths or limitations.

### Can they reproduce and sustain walking?

**Forecasting benchmark:** Fig2_self_driving.jl / fig-selfdriving. Compare prediction, sustained behavior, and repeatability, with simple reference forecasts to show what fitted dynamics add.

**Spectral constraints (2026-11-05):** stabilization_benchmark.jl. Unconstrained DMD, tuned by the same training-data criterion, also keeps the cycle in 12 of 13 cells and never blows up; removing the constraint from the selected fits leaves 11 of 13. The two-trial Fig_stabilization example is withdrawn from the manuscript; the unconstrained result is a bar in fig-selfdriving B.


### Can they forecast the short-time response?

Walking's future has a predictable part (responses to disturbances already received) and an unpredictable part (disturbances still to come). Test how much of the predictable part each model forecasts, against a ceiling known from ground truth.

**Figure:** Fig_predictable.jl / fig-predictable (slip_predictable_truth.jl in the SLIP.jl environment → predictable_response.jl). Kicked SLIP walks of Fig 1A, 1% measurement noise, 60 or 300 training steps, rollout validation on an independent walk, 300 held-out starts; skill against the mean cycle, ceiling = noise-free continuation of the true state.

- The predictable horizon is about a stride: with feedback, ceiling skill 0.73 over the first stride, 0.32 over strides 1–3.
- Full state, 60 steps: share of attainable skill, first stride / strides 1–3: DMD 82% / 27%, ESN 89% / 63%; with 300 steps DMD 86%, ESN 93% beyond the first stride. Open loop: similar.
- Height only: ESN stays near the ceiling; DMD falls below the mean cycle after about 1.5 strides. Partial observation is a DMD weakness (R4).
- The ESN is the more accurate forecaster here; the case for linear models rests on the combination of requirements.
- The large-kick coupled-FHN figure and the small-kick comparison are supplementary (suppfig-large-kicks, suppfig-small-kicks).

### What is the overall comparison?

Use the scorecard to assess the requirements together. The case for linear models rests on combined practical strengths, with explicit recognition of cases where a nonlinear model predicts better. Match claims to the systems, data, and fitting procedures actually tested.

## 5. Why linear models are a strong foundation for this broader goal

Explain “linear” accessibly: dynamics can be linear in an expanded representation, such as a history of movements, while producing rich oscillatory behavior in the measured signals.

### They make fitting and model checking tractable

- For a fixed representation, transparent regression provides reproducible fits at modest computational cost.
- Spectral constraints help impose desired sustained behavior; refer back to the rollout example.
- Some measurement-noise biases can be analyzed and corrected; refer back to the Floquet-map example.
- Representation, model order, and data coverage still require validation.

### They let us ask richer questions about the response

This is the natural home for the companion paper’s non-normality, transient-growth, resolvent, and pseudospectral analyses. Introduce each through a gait question:

- **Transient growth:** which disturbances can produce the largest excursions over the next few strides, and when?
- **Response directions and coordination:** which combinations of movements are amplified, and where does the response appear?
- **Resolvent gains:** which temporal patterns of repeated forcing produce large responses in the specified model?
- **Pseudospectra:** how sensitive are spectral conclusions to changes or uncertainty in the fitted operator?

Use one coherent, validated example rather than a gallery of spectral plots. Connect calculations to observed movements and explicit perturbations. Gain over an entire delay history is not automatically the gain from an instantaneous physical kick. Stabilized neutral modes also require an explicit convention for resolvent evaluation.

**Reader’s takeaway:** linear models provide connected tools for predicting and examining gait response, extending beyond a stability number. Physical meaning comes from the modeled inputs, outputs, and validation, not simply from having a matrix to analyze.

## 6. What does this mean for human gait studies?

Return to the questions of response and recovery.

- Explain what ordinary walking records establish and what requires informative disturbances or task changes.
- Use Fig5_gait_variability.jl / fig-gait-mean to show how strongly the usual gait pattern can dominate a fitted model. Reproducing or distinguishing waveforms differs from learning recovery.
- Describe limited deviation-prediction skill in the tested short recordings as a finding about those data and analyses, not a universal claim that human gait is unpredictable.
- A model fitted under active control describes the combined observed system. Separating mechanics from feedback or attributing behavior to a physiological mechanism requires additional experimental information.
- Frame applications as questions to validate: whether response features differ between people or conditions, change with training, a device, or impairment, or predict recovery in an independent perturbation task. Clinical populations are one such context, not the paper's motivation.

## 7. Conclusion

The goal is to understand and predict how walking responds, compensates, and recovers. Asymptotic stability contributes to that description but is not an adequate primary target on its own. Models should be judged against the broader requirements of this task. Linear models currently offer a particularly useful combination of prediction, reproducibility, tractability, and dynamical analysis, provided their responses are validated on relevant data.

## Manuscript status and figure work remaining

- The narrative order is now implemented in index.qmd: the conceptual and practical Floquet critiques appear together early, and model requirements precede data requirements.
- The piecewise/Floquet challenge figure remains in the main narrative. Any later simplification should preserve that argument.
- Consolidate forecasting, stabilization, and recovery figures rather than adding a main figure for every diagnostic.
- Complete a validated linear-response example before adding resolvent or pseudospectral claims.
- The recovery section now tests forecasts of the predictable short-time response (fig-predictable); the larger-kick and small-kick figures are supplementary.
- Regenerate the manuscript whenever narrative or figure assets change. The original detailed outline remains in archive/OUTLINE_before_response_reorganization.md for historical evidence and unfinished analyses.

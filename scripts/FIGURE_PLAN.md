# Figure organization and next experiments

See ../OUTLINE.md for the active narrative. Filenames keep their historical
numbers; use Quarto IDs instead of treating filenames as display order.

The 2026-11-04 outline restores the clinical argument: why asymptotic stability
is an inadequate primary target, what a useful gait model must capture, and why
linear models offer a strong practical foundation. The manuscript now follows
that order, with the larger-kick experiment in the main text and the earlier
small-kick comparison in the supplement.

| Narrative location | Script / ID | Current status |
|---|---|---|
| Why asymptotic stability misses important recovery behavior | Fig4_nonnormality.jl / fig-nonnormal | Existing ground truth; lead with contrasting responses |
| Practical Floquet and piecewise-model challenges | Fig3_floquet_maps.jl / fig-floquet-maps | Main text immediately after the conceptual critique |
| Data needed to meet the model requirements | Fig1_learnability.jl | Optional supporting experiment; preliminary coverage diagnostic only |
| Predictive benchmark | Fig2_self_driving.jl / fig-selfdriving | Existing comparison |
| Immediately after R2 | stabilization_data.jl → Fig_stabilization.jl / fig-stabilization | Paired spectral-constraint ablation |
| Earlier small-kick comparison (supplement) | Fig6_linear_affordances.jl / suppfig-small-kicks | Retained with explicit regime and initialization caveats |
| Substantial recovery | Fig_cfhn_large_kicks.jl / fig-large-kicks | Main-text immediate-kick DMD/ESN comparison with matched training-domain coverage |
| Human-data limits | Fig5_gait_variability.jl / fig-gait-mean | Existing mean-cycle control |

## Stabilization

Run stabilization_data.jl, then Fig_stabilization.jl.
The data script reproduces the train-selected fits for AB 91 and LF 231 and checks
the unit-circle rollout against the existing cache. It changes only the spectral
constraint (none, clip eigenvalues above one, or also pin persistent modes).
All three branches use the same selected settings and initial history.
The selection was for the constrained estimator; do not describe this as independent
optimization of three competing methods. The plotted 20-s result must show
the actual effect, including a negligible effect if that is what the data show.

## Replacement learnability experiment

Build a controlled same-cycle/different-response example before extending to FHN/SLIP.
Keep this experiment separate from the paired stabilization comparison.

1. Specify two or more systems sharing an observed periodic orbit and differing in
   transverse response. Record phase, perturbation direction and amplitude explicitly.
2. Plot identical on-cycle signals and the different true responses to a common kick.
3. Generate repeated passive and perturbed training datasets at matched sample budgets.
   State process and measurement noise separately and preserve trajectory boundaries.
4. Divide complete trajectories into training, validation and independent test sets.
   Select delay/rank/regularization without access to the displayed test response.
5. Test finite-time trajectories after new kicks at multiple phases/directions.
   Use one fixed coordinate scaling and test ensemble across data budgets.
6. Show prediction error relative to the true response and a declared reference.
   Summarize uncertainty across independent datasets; model seeds are a separate source
   of variation. Include failures instead of dropping them from the aggregate.
7. Only then decide whether a data-size panel adds useful evidence. Do not require
   monotonic convergence or make multiplier recovery the primary endpoint.

The existing coverage cache and matched_data_return.jl results remain available.
They do not implement this protocol. The manuscript discusses the data requirement
without a visible planning callout; this optional experiment is not presented as
completed evidence.

## Next figure edits

- Reduce the opening figure around physical response traces, with phase shifts
  distinguished from transverse deviations.
- Keep the Floquet estimation and piecewise-model challenges together in the main
  narrative. The earlier automatic main/supplement split is withdrawn. Refer back
  to the noise-bias correction when discussing advantages of linear estimation.
- Foreground held-out response curves in Fig6_linear_affordances.jl; consider moving
  its spectrum panel to the supplement.
- Add a response-operator validation example connecting fitted finite-time propagation
  to actual observed perturbations. Whole-signal operator norm alone is insufficient.

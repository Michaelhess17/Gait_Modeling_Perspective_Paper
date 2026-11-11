# Coupled-FHN trajectories underlying Figure 4A

Historical sharp-regime diagnostic notes. The superseded rendered figures were
removed during cleanup; the scripts and caches remain available for regeneration.

These plots use the exact cached predictions scored in Figure 4A:
one Hankel-DMD fit, four ESN seeds, and three neural-ODE seeds,
each evaluated from four perturbed starts. No refitting or phase alignment was applied.
Every per-start cycle-RMS distance was checked against the figure's scoring cache
(absolute tolerance 1e-8).

Each start has two plots:

- **State trajectories:** all four raw state variables, model versus truth.
- **Transient detail:** differences between the two units (v₁ − v₂ and w₁ − w₂),
  and distance to the true cycle in the standardized four-dimensional state.
  Thin distance curves are instantaneous; connected dots are RMS over each cycle.
  Axes use the same vertical scale across model columns within a row.

Black is truth; colours identify models and line styles distinguish seeds.
The grey interval is observed warm-up, beginning at the perturbation.
Time zero is autonomous release after 170 samples (about 1.5 cycles); there are
110 samples per cycle. The first Figure 4A score is the RMS of the first complete
autonomous cycle, not the instantaneous deviation at the kick.
Here those five RMS values are plotted at their bin centres (0.5 through 4.5).

## What the comparison shows

The large oscillations can look accurate while the much smaller desynchronization
fails to disappear. In start 1, DMD captures some decay but retains excess
desynchronization and cycle-shape error; the ESN fits closely track the large
oscillations while their inter-unit differences persist. Neural-ODE seeds also
show more variation in timing and waveform.

Across the four starts, median distance to the true cycle in the first and fifth
autonomous cycles is:

| Model | First cycle RMS | Fifth cycle RMS |
|---|---:|---:|
| Truth | 0.136 | 0.030 |
| Hankel-DMD | 0.161 | 0.107 |
| ESN, range across seed medians | 0.142–0.177 | 0.106–0.132 |
| Neural ODE, range across seed medians | 0.217–0.291 | 0.212–0.297 |

Distances are in the same SD units as Figure 4A. These values quantify return
to the true cycle, including errors in the model's cycle. They do not isolate
the response relative to each model's own unperturbed trajectory. Thus saying
the models capture “none of the transient” is too strong, particularly for DMD.
The failure is in accurately completing the return as the true deviation gets small.

## Reproduction

Run `julia --project=. scripts/cfhn_trajectory_diagnostics_data.jl`,
then `julia --project=. scripts/Fig_cfhn_trajectory_diagnostics.jl`.
Each plot is also saved as PDF and SVG in this directory.

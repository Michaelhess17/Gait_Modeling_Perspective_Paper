| Model | R1 forecast | R2 keeps the cycle | R3 seed dependence | R4 short, partial records | R5 multipliers | R6 between people |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| Fourier in phase | 0.32 (62%) | 92% · 100% · 100% | no seed | 0.72 · 77% · 22 s | 0.04 | 0.73 |
| phase-binned linear | 0.41 (—) | cannot run freely | no seed | 0.95 · — · 1.5 s | 0.05 | 0.89 |
| Hankel-DMD (time-invariant) | 0.62 (27%) | 92% · 100% · 98% | no seed | 0.72 · −49% · 11 s | 0.55 | 0.84 |
| ESN | 0.55 (63%) | 60% · 100% · 100% | 0.12 · 9/13 | 0.78 · 77% · 23 s | 0.18 | 0.15 |
| LSTM | 1.08 (—) | 92% · — · — | 1.06 · 2/13 | 1.17 · — · 65 s (GPU) | — | 0.13 |
| neural ODE | 0.60 (46%) | 42% · 100% · — | 0.15 · 9/13 | 0.84 · −14% · 38 s (GPU) | 0.17 | 0.69 |
| SSMLearnPy | 0.86 (24%) | 62% · 66% · 77% | no seed | 1.33 · 74% · 14 s | 0.47 | 0.18 |

: Headline numbers for each requirement, in the settings tested here (—, not tested). R1: remaining variance of the running deviations one eighth of a stride ahead, 150 training strides (median over trials; 1, no better than the average stride); in parentheses, the share of the attainable variance reduction over strides 1–3 of the SLIP walker with feedback, 60 training steps (@fig-running, @fig-predictable). R2: share of free runs that keep the cycle for 30 cycles in the benchmark · on the SLIP walker · on the running whole signal (@fig-selfdriving B). R3: squared difference between forecasts from different seeds, relative to the deviation's variance · benchmark cells in which the seeds end in different fates (@fig-reproducible). R4: remaining variance of the running deviations with 16 training strides · the SLIP share with the height alone · seconds to choose the settings and fit one running trial (@fig-realistic). R5: distance of the three leading per-cycle Floquet multipliers from the truth, 64 cycles of a simulated linear time-periodic system (@fig-recovery). R6: share of a per-trial quantity's variance between runners, intraclass correlation (@fig-heterogeneity A). {#tbl-scorecard tbl-colwidths="[18,14,17,15,18,9,9]"}

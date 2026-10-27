#!/bin/sh
# Phase 1: DMD, ESN (tune + read) and the neural-ODE hyperparameter search.
# Phase 2: final neural-ODE seeds with the best searched setting per case.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
CASES="cfhn:excited cfhn:realistic sl:excited sl:realistic"
{
  for c in $CASES; do s=${c%%:*}; r=${c##*:}
    echo "$s $r nodesearch:1"; for i in 2 3 4 5 6 7 8 9 10; do echo "$s $r nodesearch:$i"; done
    echo "$s $r esn"; echo "$s $r dmd"
  done
} | xargs -P 30 -L 1 sh -c 'julia --project=. scripts/floquet_tuned.jl $0 $1 $2 > logs/ft_$0_$1_$(echo $2 | tr -d :).log 2>&1'
{
  for c in $CASES; do s=${c%%:*}; r=${c##*:}
    for sd in 1 2 3 4; do echo "$s $r nodefinal:$sd"; done
  done
} | xargs -P 16 -L 1 sh -c 'julia --project=. scripts/floquet_tuned.jl $0 $1 $2 > logs/ft_$0_$1_$(echo $2 | tr -d :).log 2>&1'
echo done > logs/ft_ALL_DONE

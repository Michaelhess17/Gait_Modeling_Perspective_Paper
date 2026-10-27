#!/bin/sh
# Waits for run_floquet_tuned.sh, then runs the decay-data version.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
while [ ! -f logs/ft_ALL_DONE ]; do sleep 60; done
{
  for s in cfhn sl; do
    for i in 1 2 3 4 5 6 7 8 9 10; do echo "$s nodesearch:$i"; done
    echo "$s esn"; echo "$s dmd"
    for r in excited realistic; do echo "$s dmdfate:$r"; done
  done
} | xargs -P 30 -L 1 sh -c 'julia --project=. scripts/floquet_decays.jl $0 $1 > logs/fd_$0_$(echo $1 | tr -d :).log 2>&1'
{
  for s in cfhn sl; do for sd in 1 2 3 4; do echo "$s nodefinal:$sd"; done; done
} | xargs -P 8 -L 1 sh -c 'julia --project=. scripts/floquet_decays.jl $0 $1 > logs/fd_$0_$(echo $1 | tr -d :).log 2>&1'
echo done > logs/fd_ALL_DONE

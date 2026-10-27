#!/bin/sh
# SLIP off-cycle comparison for Fig 6A: train-only searches (floquet_decays.jl), then offcycle_v2.jl fits.
# Needs cache/truth/slip_decays.jls (scripts/slip_decays.jl, SLIP.jl environment).
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
{
  for i in 1 2 3 4 5 6 7 8 9 10; do echo "slip nodesearch:$i"; done
  echo "slip esn"; echo "slip dmd"
} | xargs -P 12 -L 1 sh -c 'julia --project=. scripts/floquet_decays.jl $0 $1 > logs/fd_$0_$(echo $1 | tr -d :).log 2>&1'
{
  echo "slip dmd 1"; for sd in 1 2 3 4; do echo "slip esn $sd"; done; for sd in 1 2 3; do echo "slip node $sd"; done
} | xargs -P 8 -L 1 sh -c 'julia --project=. scripts/offcycle_v2.jl $0 $1 $2 > logs/offv2_$0_$1_$2.log 2>&1'
echo done > logs/slip_offcycle_DONE

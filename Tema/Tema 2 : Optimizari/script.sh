#!/usr/bin/env bash

function run() {
    perf stat -o "$1_$2" -e instructions,cycles,branches,branch-misses,cache-references,cache-misses ./$1 $2
}

declare -a val=(400 600 800 1000 1200 1400 1600)
declare -a run_name=("tema2_neopt_gcc" "tema2_opt_m_gcc" "tema2_opt_f_gcc" "tema2_blas_gcc" "tema2_neopt_icc" "tema2_opt_m_icc" "tema2_opt_f_icc" "tema2_blas_icc")

for i in "${run_name[@]}"
do
    for j in "${val[@]}"
    do
        run $i $j
    done
done
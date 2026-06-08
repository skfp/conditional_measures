#!/bin/bash

output_dir="experiments_results_260608_qrr"
output_prefix="fld_misspec"
output="${output_dir}/${output_prefix}"
mc=1000

# --- true values (run once) ---
python truevalue_misspec.py --dgp 3a --alpha 0.5 --beta 0.2 --c 0.5 --c2 0.005 --xmax 30 --output ${output}
python truevalue_misspec.py --dgp 3b --mu0 0.5 --sigma0 0.2 --sigma1 0.02 --xmax 30 --output ${output}
python truevalue_misspec.py --dgp 3c --c_bell 0.05 --xmax 30 --output ${output}

# --- 3a: quadratic CQF (broken linearity in x) ---
for n in 50 100
do
  python mc_fld_misspec.py --dgp 3a --n ${n} --mc ${mc} \
    --alpha 0.5 --beta 0.2 --c 0.5 --c2 0.005 --xmax 30 --output ${output} &
done

# --- 3b: log-normal heteroscedastic (non-EFLD marginal shape) ---
for n in 50 100
do
  python mc_fld_misspec.py --dgp 3b --n ${n} --mc ${mc} \
    --mu0 0.5 --sigma0 0.2 --sigma1 0.02 --xmax 30 --output ${output} &
done

# --- 3c: non-monotone beta1 (bell-shaped) ---
for n in 50 100
do
  python mc_fld_misspec.py --dgp 3c --n ${n} --mc ${mc} \
    --c_bell 0.05 --xmax 30 --output ${output} &
done

wait

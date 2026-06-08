#!/bin/bash

output_dir="experiments_results_260608_qrr"
#output_dir="experiments_results"
output_prefix="fld"

output="${output_dir}/${output_prefix}"

mc=1000
ntaus=99

## compute true values
 python truevalue_fld.py --alpha 0.5 --beta 0.05 --c 0.2 --xmax 30 --output ${output} &
 python truevalue_fld.py --alpha 0.5 --beta 0.1 --c 0.1 --xmax 30 --output ${output} &
 python truevalue_fld.py --alpha 0.5 --beta 0.1 --c 0.5 --xmax 30 --output ${output} &
 python truevalue_fld.py --alpha 0.5 --beta 0.1 --c 1.0 --xmax 30 --output ${output} &
 python truevalue_fld.py --alpha 0.5 --beta 0.2 --c 0.1 --xmax 30 --output ${output} &
 python truevalue_fld.py --alpha 0.5 --beta 0.2 --c 0.3 --xmax 30 --output ${output} &
 python truevalue_fld.py --alpha 0.5 --beta 0.2 --c 1.0 --xmax 30 --output ${output} &
 python truevalue_fld.py --alpha 0.5 --beta 0.5 --c 0.1 --xmax 30 --output ${output} &
 python truevalue_fld.py --alpha 0.5 --beta 0.5 --c 0.5 --xmax 30 --output ${output} &
 wait

# compute monte carlo
for n in 50 100
do
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.05 --c 0.2 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.1 --c 0.1 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.1 --c 0.5 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.1 --c 1.0 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.2 --c 0.1 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.2 --c 0.3 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.2 --c 1.0 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.5 --c 0.1 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.5 --c 0.5 --xmax 30 --taus-float-type 0.1 --qrr-n-taus ${ntaus} --output ${output} &
done
wait

#params:
#( =0.5 =0.05 =0.2)
#( =0.5 =0.1 =0.1)
#( =0.5 =0.1 =0.5)
#( =0.5 =0.1 =1.0)
#( =0.5 =0.2 =0.1)
#( =0.5 =0.2 =0.3)
#( =0.5 =0.2 =1.0)
#( =0.5 =0.5 =0.1)
#( =0.5 =0.5 =0.5)


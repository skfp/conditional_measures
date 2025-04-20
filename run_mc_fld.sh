#!/bin/bash

output_dir="experiments_results"
output_prefix="fld"

output="${output_dir}/${output_prefix}"

mc=1000

# compute true values
python truevalue_fld.py --alpha 0.5 --beta 0.05 --c 0.2 --xmax 30 --output ${output} &
python truevalue_fld.py --alpha 0.5 --beta 0.1 --c 0.1 --xmax 30 --output ${output} &
python truevalue_fld.py --alpha 0.5 --beta 0.1 --c 0.5 --xmax 30 --output ${output} &
python truevalue_fld.py --alpha 0.5 --beta 0.1 --c 1.0 --xmax 30 --output ${output} &
python truevalue_fld.py --alpha 0.5 --beta 0.2 --c 0.1 --xmax 30 --output ${output} &
python truevalue_fld.py --alpha 0.5 --beta 0.2 --c 0.3 --xmax 30 --output ${output} &
python truevalue_fld.py --alpha 0.5 --beta 0.2 --c 1.0 --xmax 30 --output ${output} &
python truevalue_fld.py --alpha 0.5 --beta 0.5 --c 0.1 --xmax 30 --output ${output} &
python truevalue_fld.py --alpha 0.5 --beta 0.5 --c 0.5 --xmax 30 --output ${output} &
# compute monte carlo
for n in 50 100 500 1000
do
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.05 --c 0.2 --xmax 30 --taus-float-type 0.1 --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.1 --c 0.1 --xmax 30 --taus-float-type 0.1 --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.1 --c 0.5 --xmax 30 --taus-float-type 0.1 --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.1 --c 1.0 --xmax 30 --taus-float-type 0.1 --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.2 --c 0.1 --xmax 30 --taus-float-type 0.1 --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.2 --c 0.3 --xmax 30 --taus-float-type 0.1 --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.2 --c 1.0 --xmax 30 --taus-float-type 0.1 --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.5 --c 0.1 --xmax 30 --taus-float-type 0.1 --output ${output} &
  python mc_fld.py --n ${n} --mc ${mc} --alpha 0.5 --beta 0.5 --c 0.5 --xmax 30 --taus-float-type 0.1 --output ${output} &
done

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


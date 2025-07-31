#/bin/bash

export MKL_THREADING_LAYER=GNU

X=`pwd`
Y=test_00

mode=restart
checkpoint=$X/$Y/initial_checkpoint.pkl

p=broadwl
batch_size=6 # number of proteins total
n_rpx=6 # number of cpu cores
step=5 # number of training loops

# broadwl
# salloc -p $p -t 36:00:00 --ntasks=$batch_size --cpus-per-task=$n_rpx python $X/$Y/ConDiv.py $mode $checkpoint $step | tee -a $X/$Y.output
echo python3.7 $X/$Y/condiv2.py $mode $checkpoint $step | tee -a $X/$Y.output
python3.7 $X/$Y/condiv2.py $mode $checkpoint $step | tee -a $X/$Y.output

#!/bin/bash
sbatch --nodes=1
sbatch --gpus-per-node=1
sbatch --time=10:00
sbatch --job-name=corrective-unlearning 

# Recommended way if you want to enable gcc version 10 for the "sbatch" session 
#source /opt/rh/devtoolset-10/enable

python3 /data/andy_lee/github/corrective-unlearning-bench/src/main.py --dataset=CIFAR10 --num_classes=10 --model=resnet9 --pretrain_iters=7500 --dataset_method=poisoning --forget_set_size=2000 --unlearn_method=FlippingInfluence --unlearn_iters=1000 --data_dir='/data/andy_lee/github/corrective-unlearning-bench/files/data/' --save_dir='/data/andy_lee/github/corrective-unlearning-bench/files/logs/'

python3 /data/andy_lee/github/corrective-unlearning-bench/src/visualize.py --dirpath='/data/andy_lee/github/corrective-unlearning-bench/files/logs'
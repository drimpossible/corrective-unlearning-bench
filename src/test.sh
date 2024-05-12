#!/bin/bash
sbatch --nodes=1
sbatch --gpus-per-node=1
sbatch --time=10:00
sbatch --job-name=corrective-unlearning 

# Recommended way if you want to enable gcc version 10 for the "sbatch" session 
#source /opt/rh/devtoolset-10/enable

python3 main.py --dataset=CIFAR10 --num_classes=10 --model=resnet9 --pretrain_iters=1000 --dataset_method=poisoning --forget_set_size=2000 --deletion_size=250 --unlearn_method=FlippingInfluence --unlearn_iters=1000 --k=-1 


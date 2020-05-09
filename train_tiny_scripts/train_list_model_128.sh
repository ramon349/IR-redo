#!/bin/sh
#SBATCH -J tripletLoss
#SBATCH -p gpu 
#SBATCH -o triplet_loss_log.txt 
#SBATCH -G 1 
#SBATCH --mem 16G 
#SBATCH -t 48:0:0 
data="/home/rlcorre/IR-final-project/deep-ranking-master/tiny-imagenet-200/"
loss="list"
embdDim="128"
checkPoint="tiny_${loss}_${embdDim}"
prevState=""
scl enable rh-python36 "python3 general_model.py ${data} ${loss} ${embdDim} ${checkPoint}${prevState}"

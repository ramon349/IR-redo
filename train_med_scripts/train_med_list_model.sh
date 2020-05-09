#!/bin/sh
#SBATCH -J tripletLoss
#SBATCH -p gpu 
#SBATCH -o triplet_loss_log.txt 
#SBATCH -G 1 
#SBATCH --mem 16G 
#SBATCH -t 48:0:0 
data="/labs/sharmalab/cbir/dataset2/"
loss="list"
embdDim="128"
checkPoint="med_${loss}_${embdDim}"
prevState="./MRS_list2.ckpt"
scl enable rh-python36 "python3 general_model.py ${data} ${loss} ${embdDim} ${checkPoint}${prevState}"

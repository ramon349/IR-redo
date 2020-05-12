#!/bin/sh
#SBATCH -J list_1k_med
#SBATCH -p gpu 
#SBATCH -o list_1k_update.txt 
#SBATCH -G 1 
#SBATCH --mem 16G 
#SBATCH -t 48:0:0 
data="/labs/sharmalab/cbir/dataset2/"
loss="list"
embdDim="1000"
checkPoint="med_${loss}_${embdDim}"
prevState="/home/rlcorre/IR-final-project/IR-redo/checkpoints/MRS_tiny_list_128_0.ckpt"
scl enable rh-python36 "python3 ./src/general_model.py ${data} ${loss} ${embdDim} ${checkPoint} ${prevState}"

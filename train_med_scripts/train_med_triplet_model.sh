#!/bin/sh
#SBATCH -J triplet_med
#SBATCH -p gpu 
#SBATCH -o triplet_loss_log.txt 
#SBATCH -G 1 
#SBATCH --mem 16G 
#SBATCH -t 48:0:0 

checkPoint="/home/rlcorre/IR-final-project/IR-redo/Code/checkpoints/MRS24.ckpt"
offset="24"
scl enable rh-python36 "python3 train_med_model.py ${checkPoint} ${offset}"

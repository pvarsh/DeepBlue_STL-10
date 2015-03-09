#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l mem=16GB
#PBS -l walltime=1:00:00
#PBS -N DEEP_BLUE
#PBS -j oe
 
module purge

cd $HOME
mkdir DEEP_BLUE
cd DEEP_BLUE
mkdir results/

url=http://cims.nyu.edu/~erc399/DL_DeepBlue_a2_SIFAR-10/

wget ${url}papers/a2.pdf

cd results/
wget ${url}lua/result/model.net
cd ..

wget ${url}lua/main.lua
wget ${url}lua/augment.lua
wget ${url}lua/model_a1.lua
wget ${url}lua/model_cp.lua
wget ${url}lua/train.lua
wget ${url}lua/validate.lua
wget ${url}lua/result.lua

th result.lua

mv results/predictions.csv predictions.csv

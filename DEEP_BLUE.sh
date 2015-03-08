#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l mem=16GB
#PBS -l walltime=1:00:00
#PBS -N DEEP_BLUE
#PBS -M erc399@nyu.edu
#PBS -j oe
 
module purge

cd $HOME/A2/DL_DeepBlue_a2_SIFAR-10/lua

th doall.lua

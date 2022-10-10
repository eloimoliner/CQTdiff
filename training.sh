#!/bin/bash
#SBATCH  --time=1-23:59:59
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=cqtdiff_sc09
#SBATCH  --gres=gpu:a100:1
##SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/diffusion_summer_2022/experiments/%a/train_%j.out

#SBATCH --array=[133]

module load anaconda
source activate /scratch/work/molinee2/conda_envs/2022_torchot
module load gcc/8.4.0
export TORCH_USE_RTLD_GLOBAL=YES
#export HYDRA_FULL_ERROR=1
#export CUDA_LAUNCH_BLOCKING=1

n=1
#n=133

namerun=bwe_with_fixed_noise
name="${n}_$namerun"
iteration=`sed -n "${n} p" iteration_parameters.txt`
#
PATH_EXPERIMENT=experiments/cqt
mkdir $PATH_EXPERIMENT

#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration
python train.py model_dir="$PATH_EXPERIMENT" $iteration \
                 wandb.run_name=$name \
                 restore=True \
                 lr=2e-4 \
                diffusion_parameters.Schurn=0
                

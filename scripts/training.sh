#!/bin/bash

module load anaconda
source activate /scratch/work/molinee2/conda_envs/2022_torchot
module load gcc/8.4.0
export TORCH_USE_RTLD_GLOBAL=YES


namerun=training
#
PATH_EXPERIMENT=experiments/$n
mkdir $PATH_EXPERIMENT

#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration
python train.py model_dir="$PATH_EXPERIMENT" \
                 wandb.run_name=$namerun \
                 restore=False \
                 lr=2e-4 \
                diffusion_parameters.Schurn=5 \
                log_interval=10 \
                save_interval=10 \
                batch_size=1
                

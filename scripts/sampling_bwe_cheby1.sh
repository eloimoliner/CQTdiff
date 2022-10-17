#!/bin/bash
#SBATCH  --time=1-23:59:59
##SBATCH  --time=00:59:59
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sampling_1_guided_cqt
#SBATCH  --gres=gpu:a100:1
##SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/diffusion_summer_2022/sampling/sampling_%j.out

##SBATCH --array=[123]

module load anaconda
source activate /scratch/work/molinee2/conda_envs/2022_torchot
module load gcc/8.4.0
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

#n=$SLURM_ARRAY_TASK_ID
exp_name="cqt"

if [ "$exp_name" = "cqt" ]; then

    ckpt="cqt_weights.pt" 
fi 

n=1
namerun=sampling_bwe
name="${n}_$namerun"
iteration=`sed -n "${n} p" iteration_parameters.txt`
#

PATH_EXPERIMENT=experiments/$exp_name
echo $PATH_EXPERIMENT


audio_len=65536
#audio_len=131072
#audio_len=262144
#audio_len=1048576
#audio_len=524288
#audio_len=160531


python sample.py $iteration \
         model_dir="$PATH_EXPERIMENT" \
         architecture="unet_CQT" \
         inference.mode="bandwidth_extension" \
         inference.load.load_mode="maestro_test" \
         inference.load.seg_size=$audio_len\
         inference.load.seg_idx=10\
         inference.checkpoint=$ckpt \
         inference.bandwidth_extension.filter.type="cheby1" \
         inference.bandwidth_extension.filter.order=6 \
         inference.bandwidth_extension.filter.fc=1000 \
         inference.bandwidth_extension.filter.ripple=0.05 \
         inference.T=35 \
         extra_info=$exp_name \
         inference.exp_name=$exp_name \
         diffusion_parameters.sigma_min=1e-4 \
         diffusion_parameters.sigma_max=1 \
         diffusion_parameters.ro=13\
         diffusion_parameters.Schurn=5 \
         inference.xi=0.2\
         audio_len=$audio_len\
         inference.data_consistency=False\




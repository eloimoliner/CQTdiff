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

    n=124
    ckpt="weights-319999.pt" 

    #tn_folder="experiments/$n_tn"
    #tn_sigma_data=0.063
fi 


n=1
namerun=sampling_bwe
name="${n}_$namerun"
iteration=`sed -n "${n} p" iteration_parameters.txt`
#

PATH_EXPERIMENT=experiments/$exp_name
echo $PATH_EXPERIMENT

#Schurn=5
Schurn=5
Stmin=0
Snoise=1

T=35
#T=1

sigma_min=1e-4
sigma_max=1
ro=13

alpha=0.2
no_replace=True
tresh_grads=0
#use_dynamic_thresh=False
#dynamic_thresh_percentile=0.95

fc=1000

#order=8
#filttype="cheby1_fir"
#fir_order=600
filttype="firwin"
order=500

#audio_len=65536
#audio_len=131072
audio_len=262144
#audio_len=1048576
#audio_len=524288
#audio_len=160531
#overlap=0.1
overlap=0 #we need to substract the discarded samples to this
#overlap=0 #we need to substract the discarded samples to this
discarded_samples=0

mode="bandwidth_extension"

python sample.py $iteration \
         model_dir="$PATH_EXPERIMENT" \
         inference.mode=$mode \
         inference.checkpoint=$ckpt \
         inference.T=$T \
         extra_info=$exp_name \
         inference.exp_name=$exp_name \
         diffusion_parameters.sigma_min=$sigma_min \
         diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=$fc inference.filter.order=$order inference.filter.fir_order=$fir_order inference.filter.type=$filttype audio_len=$audio_len inference.long_sampling.overlap=$overlap inference.long_sampling.discard_samples=$discarded_samples inference.no_replace=$no_replace inference.max_thresh_grads=$tresh_grads 



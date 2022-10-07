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
#exp_name="cqt_stft_rc_pe"
#exp_name="cqt_stft_rc_zp"
#exp_name="cqt_stft_nobias"
exp_name="cqt_guidance_norep"
#exp_name="cqt"
#exp_name="cqt_deterministic"
#exp_name="cqt_gain_aug"
#exp_name="cqt_nopad"
#exp_name="sashimi"
#exp_name="stft"
if [ "$exp_name" = "sashimi" ]; then

    n=120
    ckpt="weights-129999.pt" 

    #tn_folder="experiments/$n_tn"
    #tn_sigma_data=0.063
fi 
if [ "$exp_name" = "cqt_nopad" ]; then

    n=132
    ckpt="weights-619999.pt" 

    #tn_folder="experiments/$n_tn"
    #tn_sigma_data=0.063
fi 
if [ "$exp_name" = "cqt_gain_aug" ]; then

    n=130
    ckpt="weights-469999.pt" 

    #tn_folder="experiments/$n_tn"
    #tn_sigma_data=0.063
fi 
if [ "$exp_name" = "cqt_deterministic" ]; then

    n=124
    ckpt="weights-319999.pt" 

    #tn_folder="experiments/$n_tn"
    #tn_sigma_data=0.063
fi 
if [ "$exp_name" = "cqt_guidance_norep" ]; then

    n=124
    ckpt="weights-319999.pt" 

    #tn_folder="experiments/$n_tn"
    #tn_sigma_data=0.063
fi 
if [ "$exp_name" = "cqt" ]; then

    n=124
    ckpt="weights-319999.pt" 

    #tn_folder="experiments/$n_tn"
    #tn_sigma_data=0.063
fi 
if [ "$exp_name" = "stft" ]; then

    n=125
    ckpt="weights-289999.pt" 

    #tn_folder="experiments/$n_tn"
    #tn_sigma_data=0.063
fi 


if [ "$exp_name" = "cqt_stft_nobias" ]; then

    n=123
    ckpt="weights-169999.pt" 

    n_tn=125
    ckpt_tn="weights-299999.pt"
    arch_tn="unet_stft_nobias"

    tn_folder="experiments/$n_tn"
    tn_sigma_data=0.063

fi 
if [ "$exp_name" = "cqt_stft_rc_pe" ]; then

    n=119
    ckpt="weights-139999.pt" 

    n_tn=121
    ckpt_tn="weights-399999.pt"
    arch_tn="unet_STFT_102_rc_pe"

    tn_folder="experiments/$n_tn"
    tn_sigma_data=0.063
fi 
if [ "$exp_name" = "cqt_stft_rc_zp" ]; then

    n=117
    ckpt="weights-249999.pt" 

    n_tn=118
    ckpt_tn="weights-699999.pt"
    arch_tn="unet_STFT_102_mirror_rc_zp"

    tn_folder="experiments/$n_tn"
    tn_sigma_data=0.063
fi 


namerun=dfdaf
name="${n}_$namerun"
iteration=`sed -n "${n} p" iteration_parameters.txt`
#

PATH_EXPERIMENT=experiments/${n}
echo $PATH_EXPERIMENT

#Schurn=5
Schurn=5
Stmin=0
Snoise=1

T=70
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

#python sampling_test_bwe.py model_dir="$PATH_EXPERIMENT" $iteration wandb.project="sample_diffusion"  wandb.run_name=$name inference.checkpoint=$ckpt inference.T=$T  extra_info=$exp_name  inference.exp_name=$exp_name STN.model_tn_architecture=$arch_tn STN.model_tn_checkpoint=$ckpt_tn STN.sigma_data_tn=$tn_sigma_data STN.model_tn_dir=$tn_folder diffusion_parameters.sigma_min=$sigma_min diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=$fc inference.filter.order=100 inference.filter.fir_order=300 inference.filter.type="firwin"  stft.model.win_size=1024 stft.model.hop_size=256 unet_STFT.nlayers=[8,7,6,5,4,3]

#python sampling_test_bwe_onestep.py model_dir="$PATH_EXPERIMENT" $iteration wandb.project="sample_diffusion"  wandb.run_name=$name inference.checkpoint=$ckpt inference.T=$T  extra_info=$exp_name  inference.exp_name=$exp_name  diffusion_parameters.sigma_min=$sigma_min diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=$fc inference.filter.order=$order inference.filter.fir_order=$fir_order inference.filter.type=$filttype audio_len=$audio_len inference.long_sampling.overlap=$overlap inference.long_sampling.discard_samples=$discarded_samples inference.no_replace=$no_replace
#python sampling_test_bwe_test.py model_dir="$PATH_EXPERIMENT" $iteration wandb.project="sample_diffusion"  wandb.run_name=$name inference.checkpoint=$ckpt inference.T=$T  extra_info=$exp_name  inference.exp_name=$exp_name  diffusion_parameters.sigma_min=$sigma_min diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=$fc inference.filter.order=$order inference.filter.fir_order=$fir_order inference.filter.type=$filttype audio_len=$audio_len inference.long_sampling.overlap=$overlap inference.long_sampling.discard_samples=$discarded_samples inference.no_replace=$no_replace inference.max_thresh_grads=$tresh_grads  inference.use_dynamic_thresh=$use_dynamic_thresh inference.dynamic_thresh_percentile=$dynamic_thresh_percentile
python sampling_test_bwe_second_try.py model_dir="$PATH_EXPERIMENT" $iteration wandb.project="sample_diffusion"  wandb.run_name=$name inference.checkpoint=$ckpt inference.T=$T  extra_info=$exp_name  inference.exp_name=$exp_name  diffusion_parameters.sigma_min=$sigma_min diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=$fc inference.filter.order=$order inference.filter.fir_order=$fir_order inference.filter.type=$filttype audio_len=$audio_len inference.long_sampling.overlap=$overlap inference.long_sampling.discard_samples=$discarded_samples inference.no_replace=$no_replace inference.max_thresh_grads=$tresh_grads 


#python sampling_test_bwe_onestep.py model_dir="$PATH_EXPERIMENT" $iteration wandb.project="sample_diffusion"  wandb.run_name=$name inference.checkpoint=$ckpt inference.T=$T  extra_info=$exp_name  inference.exp_name=$exp_name  diffusion_parameters.sigma_min=$sigma_min diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=$fc inference.filter.order=8 inference.filter.ripple=0.05 inference.filter.type="cheby1_fir"  inference.filter.fir_order=100
#python sampling_test_bwe_onestep.py model_dir="$PATH_EXPERIMENT" $iteration wandb.project="sample_diffusion"  wandb.run_name=$name inference.checkpoint=$ckpt inference.T=$T  extra_info=$exp_name  inference.exp_name=$exp_name  diffusion_parameters.sigma_min=$sigma_min diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=$fc inference.filter.order=9 inference.filter.ripple=0.05 inference.filter.type="cheby1filtfilt" 

#python sampling_test_bwe_onestep.py model_dir="$PATH_EXPERIMENT" $iteration wandb.project="sample_diffusion"  wandb.run_name=$name inference.checkpoint=$ckpt inference.T=$T  extra_info=$exp_name  inference.exp_name=$exp_name  diffusion_parameters.sigma_min=$sigma_min diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=$fc inference.filter.order=8 inference.filter.type="cheby1_fir"  inference.filter.fir_order=100

#python sampling_test_bwe.py model_dir="$PATH_EXPERIMENT" $iteration wandb.project="sample_diffusion"  wandb.run_name=$name inference.checkpoint=$ckpt inference.T=$T  extra_info=$exp_name  inference.exp_name=$exp_name STN.model_tn_architecture=$arch_tn STN.model_tn_checkpoint=$ckpt_tn STN.sigma_data_tn=$tn_sigma_data STN.model_tn_dir=$tn_folder diffusion_parameters.sigma_min=$sigma_min diffusion_parameters.sigma_max=$sigma_max diffusion_parameters.ro=$ro diffusion_parameters.Schurn=$Schurn diffusion_parameters.Stmin=$Stmin  diffusion_parameters.Snoise=$Snoise inference.alpha=$alpha inference.filter.fc=1000 inference.filter.order=8 inference.filter.fir_order=300 inference.filter.type="cheby1_fir"

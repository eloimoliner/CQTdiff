import numpy as np
import os
import re
import torch
import torchaudio
import torch.nn as nn

from tqdm import tqdm
from glob import glob

#from dataset import from_path as dataset_from_path
from getters import get_sde
import time
from inference import Deterministic_Sampling_Elucidating, Stochastic_Sampling_Elucidating
import utils
import utils_plotting
import lowpass_utils
import wandb

from sde import  VE_Sde_Elucidating


class Learner:
    def __init__(
        self, model_dir, model, train_set,  optimizer, args, log=True
    ):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.step = 0
        if args.restore:
            self.restore_from_checkpoint()

        self.ema_weights = [param.clone().detach()
                            for param in self.model.parameters()]

        if args.sde_type=='VE_elucidating':
            self.diff_parameters=VE_Sde_Elucidating(args.diffusion_parameters, args.diffusion_parameters.sigma_data)
        else:
            raise NotImplementedError

        self.det_sampler=Deterministic_Sampling_Elucidating(self.model,self.diff_parameters)
        self.stoch_sampler=Stochastic_Sampling_Elucidating(self.model,self.diff_parameters)

        self.ema_rate = args.ema_rate
        self.train_set = train_set
        self.optimizer = optimizer
        self.args = args
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.scheduler_step_size, gamma=self.args.scheduler_gamma)


        self.loss_fn = nn.MSELoss()
        self.v_loss = nn.MSELoss(reduction="none")

        self.summary_writer = None
        self.n_bins = args.n_bins

        self.accumulated_losses=None
        self.accumulated_losses_sigma=None


        self.cum_grad_norms = 0
        self.device=next(self.model.parameters()).device
        #self.filters=lowpass_utils.get_random_FIR_filters(self.args.bwe.num_random_filters, mean_fc=self.args.bwe.lpf.mean_fc, std_fc=self.args.bwe.lpf.std_fc, device=self.device,sr=self.args.sample_rate)# more parameters here
        self.log=log
        if self.log:
            #report hyperparameters. add or remove here the relevant hyperparams
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("total_params: ",total_params/1e6, "M")
            config_dict = {
                  "learning_rate": self.args.lr,
                  "audio_len": self.args.audio_len,
                  "sample_rate": self.args.sample_rate,
                  "batch_size": self.args.batch_size,
                  "microbatches": self.args.microbatches,
                  "dataset": self.args.dset.name,
                  "sde_type": self.args.sde_type,
                  "architecure": self.args.architecture,
                  "num_steps": args.inference.T,
                  "ro": args.diffusion_parameters.ro,
                  "sigma_max": args.diffusion_parameters.sigma_max,
                  "sigma_min": args.diffusion_parameters.sigma_min,
                  "Schurn": args.diffusion_parameters.Schurn,
                  "Snoise": args.diffusion_parameters.Snoise,
                  "Stmin": args.diffusion_parameters.Stmin,
                  "Stmax": args.diffusion_parameters.Stmax,
                  "total_params": total_params
                   }
            wandb.init(project=args.wandb.project, entity=args.wandb.entity, config=config_dict)
            wandb.run.name=args.wandb.run_name+"_"+wandb.run.id
    
            self.first_log=True #just for logging the test table
        #self.CQTransform=utils.CQT(fmin,fbins, args.fs, args.seg_len_s_train, device=self.device, split_0_nyq=True)

        S=self.args.resample_factor
        if S>2.1 and S<2.2:
            #resampling 48k to 22.05k
            self.resample=torchaudio.transforms.Resample(160*2,147).to(self.device) 
        else:
            N=int(self.args.audio_len*S)

            self.resample=torchaudio.transforms.Resample(N,self.args.audio_len).to(self.device) 

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "step": self.step,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            'ema_weights': self.ema_weights,
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.step = state_dict["step"]
        self.ema_weights = state_dict['ema_weights']

    def save_to_checkpoint(self, filename="weights"):
        save_basename = f"{filename}-{self.step}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)

    def restore_from_checkpoint(self, checkpoint_id=None):
        try:
            if checkpoint_id is None:
                # find latest checkpoint_id
                list_weights = glob(f'{self.model_dir}/weights-*')
                id_regex = re.compile('weights-(\d*)')
                list_ids = [int(id_regex.search(weight_path).groups()[0])
                            for weight_path in list_weights]
                checkpoint_id = max(list_ids)

            checkpoint = torch.load(
                f"{self.model_dir}/weights-{checkpoint_id}.pt")
            self.load_state_dict(checkpoint)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def sample(self, num_samples):
        shape=(num_samples, self.args.audio_len)

        res_det=self.det_sampler.predict(shape,35, self.device)

        self._write_summary_sample(res_det, "deterministic")


    def train(self):
        device = self.device
        while True:
            start=time.time()
            
            loss, vectorial_loss, sigma= self.train_step()

            sigma_detach = sigma.clone().detach().cpu().numpy()
            sigma_detach = np.reshape(sigma_detach, -1)
            vectorial_loss = torch.mean(vectorial_loss, 1).cpu().numpy()
            vectorial_loss = np.reshape(vectorial_loss, -1)
            self.update_accumulated_loss(vectorial_loss, sigma_detach, True)

            if (self.step+1) % self.args.save_interval==0:
                if self.args.save_model:
                    self.save_to_checkpoint()

                #Doing the heavy logging here!
                if self.log:
                    self.sample(8)
                
            if (self.step+1) % self.args.log_interval == 0:
                #Doing all the light logging here!
                if self.log:
                    self._write_summary(self.step)

            self.step += 1
            end=time.time()

            print("Step: ",self.step,", Loss: ",loss.item(),", Time: ",end-start)

    def get_data_batch():
        #get one batch of data from the dataset and resample it (if necessary)
        y=self.train_set.next()    
        y=y.to(self.device)

        if self.args.resample_factor!=1:
            y=self.resample(y)

        return y

    def train_step(self):

        for param in self.model.parameters():
            param.grad = None

        audio = get_data_batch()

        N, T = audio.shape
        device=audio.device
        
        #sample a random batch of noise levels
        sigma=self.diff_parameters.sample_ptrain_alt(N)
        sigma=torch.Tensor(sigma).to(audio.device)
        sigma=sigma.unsqueeze(-1)
            

        #compute the scaling parameters
        cskip=self.diff_parameters.cskip(sigma)
        cout=self.diff_parameters.cout(sigma)
        cin=self.diff_parameters.cin(sigma)
        cnoise=self.diff_parameters.cnoise(sigma)

        #sample the noise instance
        noise = torch.randn_like(audio)*sigma

        #apply the NN
        estimate=self.model(cin*(audio+noise),cnoise) 

        #target as in Karras et al. "Elucidating..." Eq. 8
        target=(1/cout)*(audio-cskip*(audio+noise))
        
        #apply the L2 loss
        loss = self.loss_fn(estimate, target) 

        loss.backward()

        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.update_ema_weights()

        vectorial_loss = self.v_loss(estimate, target).detach()

        self.cum_grad_norms += self.grad_norm

        return loss, vectorial_loss, sigma


    def _write_summary_sample(self,res, string):
        #print(res.shape)
        spec_sample=utils_plotting.plot_spectrogram_from_raw_audio(res, self.args.stft)
        wandb.log({"spec_sample_"+str(string): spec_sample}, step=self.step)

        audio_path=utils_plotting.write_audio_file(res, self.args.sample_rate)
        wandb.log({"audio_sample_"+str(string): wandb.Audio(audio_path, sample_rate=self.args.sample_rate)},step=self.step)

        spec_sample=utils_plotting.plot_CQT_from_raw_audio(res, self.args)
        wandb.log({"CQT_sample_"+str(string): spec_sample}, step=self.step)

    def _write_summary(self, step):
        #just logging a plot of the loss and a couple of scalars

        sigma_max=self.diff_parameters.sigma_max
        sigma_min=self.diff_parameters.sigma_min
        quantized_sigma_values=self.diff_parameters.create_schedule(self.n_bins)
        ro=self.diff_parameters.ro
        sigma=self.accumulated_losses_sigma
        quantized_sigma=(sigma**(1/ro) -sigma_max**(1/ro))*(self.n_bins-1)/(sigma_min**(1/ro) -sigma_max**(1/ro))
        quantized_sigma.astype(int)

        num_elems_in_bins = np.zeros(self.n_bins)
        sum_loss_in_bins = np.zeros(self.n_bins)

        for k in range(len(quantized_sigma)):
            i_bin=int(quantized_sigma[k])
            num_elems_in_bins[i_bin]+=1
            sum_loss_in_bins[i_bin]+=self.accumulated_losses[k]
        
        #write a fancy plot to log in wandb
        figure=utils_plotting.plot_loss_by_sigma_train(sum_loss_in_bins, num_elems_in_bins, quantized_sigma_values[:-1])
        wandb.log({"loss_dependent_on_sigma": figure}, step=self.step)

        averaged_loss=np.mean(self.accumulated_losses)
        wandb.log({"averaged_loss": averaged_loss},step=self.step)

        mean_grad_norms = self.cum_grad_norms /num_elems_in_bins.sum() * self.args.batch_size

        wandb.log({"mean_grad_norm": mean_grad_norms},step=self.step, commit=True)

        self.cum_grad_norms = 0
        self.accumulated_losses=None
        self.accumulated_losses_sigma=None


    def update_accumulated_loss(self, vectorial_loss, sigma_array, isTrain):

        if (self.accumulated_losses is None) and (self.accumulated_losses_sigma is None):
             self.accumulated_losses= vectorial_loss
             self.accumulated_losses_sigma= sigma_array
        else:
             assert not((self.accumulated_losses is None) or (self.accumulated_losses_sigma is None)), "either both or none of them should be None"
             self.accumulated_losses=np.concatenate( (self.accumulated_losses,vectorial_loss), axis=0)
             self.accumulated_losses_sigma=np.concatenate( (self.accumulated_losses_sigma,sigma_array), axis=0)

    def update_ema_weights(self):
        for ema_param, param in zip(self.ema_weights, self.model.parameters()):
            if param.requires_grad:
                ema_param -= (1 - self.ema_rate) * (ema_param - param.detach())
            else:
                ema_param=param




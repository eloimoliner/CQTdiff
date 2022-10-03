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

from decSTN_pytorch import decSTNsingle
from torch_audiomentations import Compose, Gain, PolarityInversion, PitchShift, Identity

#def _nested_map(struct, map_fn):
#    if isinstance(struct, tuple):
#        return tuple(_nested_map(x, map_fn) for x in struct)
#    if isinstance(struct, list):
#        return [_nested_map(x, map_fn) for x in struct]
#    if isinstance(struct, dict):
#        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
#    return map_fn(struct)


class Learner:
    def __init__(
        self, model_dir, model, train_set, test_set, optimizer, args, log=True
    ):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.step = 0
        if args.restore:
            self.restore_from_checkpoint()

        self.ema_weights = [param.clone().detach()
                            for param in self.model.parameters()]

        self.diff_parameters = get_sde(args.sde_type, args.diffusion_parameters, args.diffusion_parameters.sigma_data)

        self.det_sampler=Deterministic_Sampling_Elucidating(self.model,self.diff_parameters)
        self.stoch_sampler=Stochastic_Sampling_Elucidating(self.model,self.diff_parameters)

        self.ema_rate = args.ema_rate
        self.train_set = train_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.args = args
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.scheduler_step_size, gamma=self.args.scheduler_gamma)

        self.is_master = True

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
        transf=[]
        if self.args.augmentations.rev_polarity:
            transf.append(
                PolarityInversion(
                        mode="per_example",
                        p=0.5)
                )
        if self.args.augmentations.pitch_shift.use:
            transf.append(
                PitchShift(
                        min_transpose_semitones = self.args.augmentations.pitch_shift.min_semitones,
                        max_transpose_semitones= self.args.augmentations.pitch_shift.max_semitones,
                        sample_rate=int(self.args.sample_rate),
                        mode="per_example",
                        p=0.9)
                )
        if self.args.augmentations.gain.use:
            transf.append(
                Gain(
                        min_gain_in_db = self.args.augmentations.gain.min_db,
                        max_gain_in_db= self.args.augmentations.gain.max_db,
                        sample_rate=int(self.args.sample_rate),
                        mode="per_example",
                        p=0.9)
                )
        if len(transf)==0:
            transf.append(Identity())

        self.apply_augmentation = Compose(
            transforms=transf
            ) 
        S=self.args.resample_factor
        if S>2.1 and S<2.2:
            #resampling 48k to 22.05k
            self.resample=torchaudio.transforms.Resample(160*2,147).to(self.device) #I use 2**12 as an arbitrary number, as we don't care about the sampling frequency of the latents
        else:
            N=int(self.args.audio_len*S)

            self.resample=torchaudio.transforms.Resample(N,self.args.audio_len).to(self.device) #I use 2**12 as an arbitrary number, as we don't care about the sampling frequency of the latents

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
        #res_stoch=self.stoch_sampler.predict(shape,200, self.device)
        res_det=self.det_sampler.predict(shape,35, self.device)

        self._write_summary_sample(res_det, "deterministic")
        #self._write_summary_sample(res_stoch, "stochastic")

        if self.args.normalization.use_mu_law:
            #res_stoch=self.mu_law_inverse(res_stoch,self.args.normalization.mu)
            res_det=self.mu_law_inverse(res_det,self.args.normalization.mu)
        if self.args.normalization.apply_pre_emph:
            #res_stoch=self.de_emphasis(res_stoch,self.args.normalization.bcoeffs)/100 #normalize to avoid clipping
            res_det=self.de_emphasis(res_det,self.args.normalization.bcoeffs) #normalize to avoid clipping
            self._write_summary_sample(res_det, "deterministic_deemph")
        
        #res=res.flatten()
        #if self.args.pre_emph.use_pre_emph:        
        #    res=self.apply_de_emph(res,self.args.pre_emph.bcoeffs)

        #self._write_summary_sample(res_stoch, "stochastic")

        #return res


    def do_stft(self,x):
        win_size=self.args.stft.win_size
        hop_size=self.args.stft.hop_size
        window=torch.hamming_window(window_length=self.args.stft.win_size)
        window=window.to(x.device)
        x=torch.cat((x, torch.zeros(x.shape[0],win_size).to(x.device)),1)
        stft_signal_noisy=torch.stft(x, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
        stft_signal_noisy=stft_signal_noisy.permute(0,2,1,3)
       
        return stft_signal_noisy.unsqueeze(1)

    def do_istft(self,x):
        x=x.squeeze(1)
        win_size=self.args.stft.win_size
        hop_size=self.args.stft.hop_size
        window=torch.hamming_window(window_length=win_size) #this is slow! consider optimizing
        window=window.to(x.device)
        x=x.permute(0,2,1,3)
        pred_time=torch.istft(x, win_size, hop_length=hop_size,  window=window, center=False, return_complex=False)
        return pred_time

    def train(self):
        device = self.device
        while True:
            start=time.time()
            
            #features = _nested_map(
            #    features,
            #    lambda x: x.to(device) if isinstance(
            #        x, torch.Tensor) else x,
            #)
            loss, vectorial_loss, sigma= self.train_step()


            sigma_detach = sigma.clone().detach().cpu().numpy()
            sigma_detach = np.reshape(sigma_detach, -1)
            vectorial_loss = torch.mean(vectorial_loss, 1).cpu().numpy()
            vectorial_loss = np.reshape(vectorial_loss, -1)
            self.update_accumulated_loss(vectorial_loss, sigma_detach, True)


            if (self.step+1) % self.args.save_interval==0:
                #Doing all the heavy computation for logging here!
                if self.args.save_model:
                    self.save_to_checkpoint()
                if self.log:
                    self.sample(8)
                    #if not(self.test_set is None):
                    #    self.test()
                
            if (self.step+1) % self.args.log_interval == 0:

                if self.log:
                    self._write_summary(self.step)

            self.step += 1
            end=time.time()

            print("Step: ",self.step,", Loss: ",loss.item(),", Time: ",end-start)

    def test(self):
        print("testing")
        #t = torch.ones(1,1,1,1,1, device=self.device) #setting t=1 
        #noise = torch.randn(1,1, self.T, self.F,2, device=self.device)
        #t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min
        #sigma = self.sde.sigma(t)
        #sigma=sigma.squeeze(2).squeeze(2).squeeze(2)
        #with torch.no_grad():
        #    pred=self.model(noise,sigma)

        #self._write_summary_A_matrix(self.model.A,self.step, "sample_first")
        ts=self.diff_parameters.create_schedule(10)

        average_losses = np.zeros(len(ts))
        average_snr = np.zeros(len(ts))
        average_snr_out = np.zeros(len(ts))

        dic={}
        for i, y in enumerate(tqdm(self.test_set)):
            for j,t_value in enumerate(ts):
                yF, noisy_audio, predicted, loss, snr, snr_out =self.test_step(y, t_value)
                average_losses[j]+=loss.item()
                average_snr[j]+=snr.item()
                average_snr_out[j]+=snr_out.item()
                #print(j, i, loss.item())
                
                #if i==0:
                #    YF, NA, PR=yF, noisy_audio, predicted
                #    #print(len(Amatrix))
                #    #stacked_Amatrix+=Amatrix
                #elif i<6:
                #    YF=torch.cat((YF,yF), dim=0)
                #    NA=torch.cat((NA,noisy_audio), dim=0)
                #    PR=torch.cat((PR,predicted), dim=0)
                #    #stacked_Amatrix+=Amatrix
    
                #dic[t_value]=[YF, NA, PR]


        average_losses= average_losses/len(self.test_set)
        average_snr = 10*np.log10(average_snr /len(self.test_set))
        average_snr_out = 10*np.log10(average_snr_out /len(self.test_set))

        dic=None
        
        self._write_summary_test_set(dic, ts, average_losses, average_snr, average_snr_out)


    def _write_summary_test_set(self, dic, t, average_losses, average_snr, average_snr_out):
        

        figure=utils_plotting.plot_loss_by_sigma_test(average_losses, t)
        wandb.log({"loss_dependent_on_sigma_test": figure}, step=self.step)

        figure=utils_plotting.plot_loss_by_sigma_test_snr(average_snr, average_snr_out, t)
        wandb.log({"snr_dependent_on_sigma_test": figure}, step=self.step)

        self.first_log=False
        if self.first_log:
            columns=["step","t", "clean", "noisy", "pred"]
            self.test_table = wandb.Table(columns=columns)
    
            for t_value in dic.keys():
    
                clean=dic[t_value][0]
                noisy=dic[t_value][1]
                pred=dic[t_value][2]
    
                #wandb.log({"spec_noisy"+str(t_value): spec_noisy}, step=self.step)
                path_to_plotly_html = "./clean"+str(t_value)+".html"
                spec_error=utils_plotting.plot_spectrogram_from_raw_audio(pred, self.args.stft)
                spec_error.write_html(path_to_plotly_html, auto_play = True)
                html_error=wandb.Html(path_to_plotly_html)
                #waindb.log({"spec_pred_error"+str(t_value): spec_error}, step=self.step)
    
                path_to_plotly_html = "./clean"+str(t_value)+".html"
                spec_clean=utils_plotting.plot_spectrogram_from_raw_audio(clean,self.args.stft )
                spec_clean.write_html(path_to_plotly_html, auto_play = True)
                html_clean=wandb.Html(path_to_plotly_html)
                
                #wandb.log({"spec_clean"+str(t_value): spec_clean}, step=self.step)
                
                path_to_plotly_html = "./clean"+str(t_value)+".html"
                spec_noisy=utils_plotting.plot_spectrogram_from_raw_audio(noisy, self.args.stft)
                spec_noisy.write_html(path_to_plotly_html, auto_play = True)
                html_noisy=wandb.Html(path_to_plotly_html)
    
    
                
                self.test_table.add_data(self.step,t_value, html_clean, html_noisy, html_error)
    
            wandb.log({"table_test_set":self.test_table}, step=self.step )
            self.first_log=False

    def normalize(self, y):
        std=torch.std(y,-1).unsqueeze(-1)
        y=(self.args.diffusion_parameters.sigma_data/(std+1e-9))*y
        if torch.isnan(y).any():
            print("std", std)
            print("y_norm", y)
        return y

    def pre_emphasis(self, inp, coefs):

        weights=torch.Tensor(coefs).unsqueeze(0).unsqueeze(0).to(inp.device)
        inp=inp.unsqueeze(1) #N, 1, T
        out=torch.nn.functional.conv1d(inp, weights, padding="same")

        return out.squeeze(1)

    def de_emphasis(self,inp, coefs):

        b=torch.Tensor([1,0]).to(inp.device)
        a=torch.Tensor(coefs).to(inp.device)
        #inp=inp.cpu().numpy()[2]
    
        out = torchaudio.functional.lfilter(inp, a, b, clamp=False)
        
        return out

    def mu_law_transform(self,x, mu):
        b=np.log(1+mu)
        return torch.sign(x)*torch.log(1+mu*torch.abs(x))/b

    def mu_law_inverse(self, y, mu):
        return torch.sign(y)*((1+mu)**(torch.abs(y))-1)/mu

    def test_step(self,y, t_value):
        y=y.to(self.device)
        #normalization
        if self.args.normalization.apply_pre_emph:
            y=self.pre_emphasis(y,self.args.normalization.bcoeffs)
        if self.args.normalization.normalize_by_sigma_data:
            y=self.normalize(y)
        if self.args.STN.use_STN:
            if self.args.STN.mode=="s":
                with torch.no_grad():
                    y, ytn= decSTNsingle(y,self.args.sample_rate,self.args.STN.nwin)
            if self.args.STN.mode=="tn":
                with torch.no_grad():
                    ys, y= decSTNsingle(y,self.args.sample_rate,self.args.STN.nwin)

        if self.args.normalization.use_mu_law:
            y=self.mu_law_transform(y,self.args.normalization.mu)
             
        
        N, T = y.shape

        device=y.device

        sigma = torch.ones(N,1, device=y.device)*t_value #trying t=0.5 for instance 
        sigma=sigma.squeeze(-1)

        with torch.no_grad():
            noise = torch.randn_like(y)*sigma.unsqueeze(-1)
            noisy_audio=y+noise

            sigma=sigma.unsqueeze(-1)

            cskip=self.diff_parameters.cskip(sigma)
            cout=self.diff_parameters.cout(sigma)
            cin=self.diff_parameters.cin(sigma)
            cnoise=self.diff_parameters.cnoise(sigma)
                

            estimate=self.model(cin*(noisy_audio),cnoise) #Eq 8

            target=(1/cout)*(y-cskip*(noisy_audio))
            
            denoised= cout*estimate +cskip*noisy_audio
            
            #print(estimate.shape, target.shape)
            loss = self.loss_fn(estimate, target) #here a frequency weighting could be cool?
            #mse = self.loss_fn(denoised, y) #here a frequency weighting could be cool?
            snr= ((y**2).mean(-1)/ sigma.squeeze(-1)**2 )
            snr_out = ((y**2).mean(-1)/ ((y-denoised)**2).mean(-1) )


        return  y, noisy_audio, denoised, loss, snr, snr_out


    def train_step(self):
        for param in self.model.parameters():
            param.grad = None

        for i in range(self.args.microbatches): 
            #print(i)

            y=self.train_set.next()    
            y=y.to(self.device)

            #print(y.shape)
            if self.args.resample_factor!=1:
                y=self.resample(y)
            #print(y.shape)
            #print(torch.std(y))

            #apply augmentations

            y=y.unsqueeze(1)
            y = self.apply_augmentation(y)
            y=y.squeeze(1)
            #print("time_augment", time.time()-start_a)


            #normalize, not sure if this is the right way
            if self.args.normalization.apply_pre_emph:
                y=self.pre_emphasis(y,self.args.normalization.bcoeffs)
            if self.args.normalization.normalize_by_sigma_data:
                y=self.normalize(y)
            if self.args.STN.use_STN:
                if self.args.STN.mode=="s":
                    with torch.no_grad():
                        y, ytn= decSTNsingle(y,self.args.sample_rate,self.args.STN.nwin)
                if self.args.STN.mode=="tn":
                    with torch.no_grad():
                        ys, y= decSTNsingle(y,self.args.sample_rate,self.args.STN.nwin)
                
            
            #if self.args.normalization.use_mu_law:
            #    y=self.mu_law_transform(y,self.args.normalization.mu)
           
            
            for param in self.model.parameters():
                param.grad = None
    
            audio = y
    
            N, T = audio.shape
            device=audio.device
    
            sigma=self.diff_parameters.sample_ptrain_alt(N)
            sigma=torch.Tensor(sigma).to(audio.device)
            sigma=sigma.unsqueeze(-1)

            #t = torch.rand(N, 1, device=audio.device)
            #t = (self.sde.t_max - self.sde.t_min) * t + self.sde.t_min
                
            cskip=self.diff_parameters.cskip(sigma)
            cout=self.diff_parameters.cout(sigma)
            cin=self.diff_parameters.cin(sigma)
            cnoise=self.diff_parameters.cnoise(sigma)
            #lambda_w=self.diff_parameters.lambda_w(sigma)
    
            noise = torch.randn_like(audio)*sigma

            estimate=self.model(cin*(audio+noise),cnoise) #Eq 8
            target=(1/cout)*(audio-cskip*(audio+noise))
            
            #print(estimate.shape, target.shape)
            if self.args.use_margin:
                ind_start=int(self.args.margin_size*estimate.shape[-1])
                ind_end=int(estimate.shape[-1]-ind_start)
                loss = self.loss_fn(estimate[...,ind_start:ind_end], target[...,ind_start:ind_end]) #here a frequency weighting could be cool?
            else:
                loss = self.loss_fn(estimate, target) #here a frequency weighting could be cool?

            loss.backward()

        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        if self.is_master:
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
        
        #self.n_bins

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

        #wandb.watch(self.model, log="all")

        self.cum_grad_norms = 0
        self.accumulated_losses=None
        self.accumulated_losses_sigma=None


    def _write_test_summary(self, step):
        # Same thing for test set
        loss_in_bins_test = np.divide(
            self.sum_loss_in_bins_test, self.num_elems_in_bins_test
        )
        dic_loss_test = {}
        for k in range(self.n_bins):
            dic_loss_test["loss_bin_" + str(k)] = loss_in_bins_test[k]

        writer = self.summary_writer or SummaryWriter(
            self.model_dir, purge_step=step)
        writer.add_scalars("test/conditioned_loss", dic_loss_test, step)
        writer.flush()
        self.summary_writer = writer
        self.num_elems_in_bins_test = np.zeros(self.n_bins)
        self.sum_loss_in_bins_test = np.zeros(self.n_bins)

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




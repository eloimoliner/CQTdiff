from tqdm import tqdm
import torch
import torchaudio
import scipy.signal
import numpy as np
import src.utils.logging as utils_logging
class Sampler():

    def __init__(self, model, diff_params, args, xi=0, order=2, data_consistency=False, rid=False):

        self.model = model
        self.diff_params = diff_params
        self.xi=xi #hyperparameter for the reconstruction guidance
        self.order=order
        self.data_consistency=data_consistency #use reconstruction gudance without replacement
        self.args=args
        self.nb_steps=self.args.inference.T

        self.treshold_on_grads=args.inference.max_thresh_grads
        self.rid=rid


    def data_consistency_step_phase_retrieval(self, x_hat, y):
        """
        Data consistency step only for the phase retrieval case, we are replacing the observed magnitude
        """
        #apply replacment (valid for linear degradations)
        win_size=self.args.inference.phase_retrieval.win_size
        hop_size=self.args.inference.phase_retrieval.hop_size

        window=torch.hamming_window(window_length=win_size).to(x_hat.device)
        #print(x.shape)
        x2=torch.cat((x_hat, torch.zeros(x_hat.shape[0],win_size ).to(x_hat.device)),-1)
        X=torch.stft(x2, win_size, hop_length=hop_size,window=window,center=False,return_complex=True)
        phaseX=torch.angle(X)
        assert y.shape == phaseX.shape, y.shape+" "+phaseX.shape
        X_out=torch.polar(y, phaseX)
        X_out=torch.view_as_real(X_out)
        x_out=torch.istft(X_out, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
        #print(x_hat.shape)
        x_out=x_out[...,0:x_hat.shape[-1]]
        #print(x_hat.shape, x.shape)
        assert x_out.shape == x_hat.shape
        return x_out

    def data_consistency_step(self, x_hat, y, degradation):
        """
        Simple replacement method, used for inpainting and FIR bwe
        """
        #get reconstruction estimate
        den_rec= degradation(x_hat)     
        #apply replacment (valid for linear degradations)
        return y+x_hat-den_rec 

    def get_score_rec_guidance(self, x, y, t_i, degradation):

        x.requires_grad_()
        x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
        den_rec= degradation(x_hat) 

        if len(y.shape)==3:
            dim=(1,2)
        elif len(y.shape)==2:
            dim=1

        norm=torch.linalg.norm(y-den_rec,dim=dim, ord=2)
        
        rec_grads=torch.autograd.grad(outputs=norm,
                                      inputs=x)

        rec_grads=rec_grads[0]
        
        normguide=torch.linalg.norm(rec_grads)/self.args.audio_len**0.5
        
        #normalize scaling
        s=self.xi/(normguide*t_i+1e-6)
        
        #optionally apply a treshold to the gradients
        if self.treshold_on_grads>0:
            #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
            rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        

        score=(x_hat.detach()-x)/t_i**2

        #apply scaled guidance to the score
        score=score-s*rec_grads

        return score

    def get_score(self,x, y, t_i, degradation):
        if y==None:
            assert degradation==None
            #unconditional sampling
            with torch.no_grad():
                x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                score=(x_hat-x)/t_i**2
            return score
        else:
            if self.xi>0:
                #apply rec. guidance
                score=self.get_score_rec_guidance(x, y, t_i, degradation)
    
                #optionally apply replacement or consistency step
                if self.data_consistency:
                    #convert score to denoised estimate using Tweedie's formula
                    x_hat=score*t_i**2+x
    
                    if self.args.inference.mode=="phase_retrieval":
                        x_hat=self.data_consistency_step_phase_retrieval(x_hat,y)
                    else:
                        x_hat=self.data_consistency_step(x_hat,y, degradation)
    
                    #convert back to score
                    score=(x_hat-x)/t_i**2
    
            else:
                #denoised with replacement method
                with torch.no_grad():
                    x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                        
                    x_hat=self.data_consistency_step(x_hat,y, degradation)
        
                    score=(x_hat-x)/t_i**2
    
            return score

    def predict_unconditional(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device
    ):
        self.y=None
        self.degradation=None
        return self.predict(shape, device)

    def predict_resample(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        shape,
        degradation, #lambda function
    ):
        self.degradation=degradation 
        self.y=y
        print(shape)
        return self.predict(shape, y.device)


    def predict_conditional(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        degradation, #lambda function
    ):
        self.degradation=degradation 
        self.y=y
        return self.predict(y.shape, y.device)

    def predict(
        self,
        shape,  #observations (lowpssed signal) Tensor with shape ??
        device, #lambda function
    ):

        if self.rid:
            data_denoised=torch.zeros((self.nb_steps,shape[0], shape[1]))

        #get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)
        #sample from gaussian distribution with sigma_max variance
        x = self.diff_params.sample_prior(shape,t[0]).to(device)

        #parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma=self.diff_params.get_gamma(t).to(device)


        for i in tqdm(range(0, self.nb_steps, 1)):
            #print("sampling step ",i," from ",self.nb_steps)

            if gamma[i]==0:
                #deterministic sampling, do nothing
                t_hat=t[i] 
                x_hat=x
            else:
                #stochastic sampling
                #move timestep
                t_hat=t[i]+gamma[i]*t[i] 
                #sample noise, Snoise is 1 by default
                epsilon=torch.randn(shape).to(device)*self.diff_params.Snoise
                #add extra noise
                x_hat=x+((t_hat**2 - t[i]**2)**(1/2))*epsilon 

            score=self.get_score(x_hat, self.y, t_hat, self.degradation)    

            #d=-t_hat*((denoised-x_hat)/t_hat**2)
            d=-t_hat*score
            
            #apply second order correction
            h=t[i+1]-t_hat


            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                score=self.get_score(x_prime, self.y, t_prime, self.degradation)

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d

            if self.rid: data_denoised[i]=x
            
        if self.rid:
            return x.detach(), data_denoised.detach(), t.detach()
        else:
            return x.detach()


class SamplerPhaseRetrieval(Sampler):

    def __init__(self, model, diff_params, args, xi=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, xi, order, data_consistency, rid)
        #assert data_consistency==False
        assert xi>0

    def apply_stft(self,x):

        x2=torch.cat((x,self.zeropad ),-1)
        X=torch.stft(x2, self.win_size, hop_length=self.hop_size,window=self.window,center=False,return_complex=False)
        Y=torch.sqrt(X[...,0]**2 + X[...,1]**2)
        return Y

    def predict_pr(
        self,
        y
        ):

        self.win_size=self.args.inference.phase_retrieval.win_size
        self.hop_size=self.args.inference.phase_retrieval.hop_size

        print(y.shape)
        self.zeropad=torch.zeros(y.shape[0],self.win_size ).to(y.device)
        self.window=torch.hamming_window(window_length=self.win_size).to(y.device)

        degradation=lambda x: self.apply_stft(x)

        return self.predict_resample(y, (y.shape[0], self.args.audio_len), degradation)
class SamplerCompSens(Sampler):

    def __init__(self, model, diff_params, args, xi=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, xi, order, data_consistency, rid)
        assert data_consistency==False
        assert xi>0

    def apply_mask(self, x):
        return self.mask*x

    def predict_compsens(
        self,
        y_masked,
        mask
        ):

        self.mask=mask.to(y_masked.device)

        degradation=lambda x: self.apply_mask(x)

        return self.predict_conditional(y_masked, degradation)

class SamplerDeclipping(Sampler):

    def __init__(self, model, diff_params, args, xi=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, xi, order, data_consistency, rid)
        assert data_consistency==False
        assert xi>0

    def apply_clip(self,x):
        x_hat=torch.clip(x,min=-self.clip_value, max=self.clip_value)
        return x_hat

    def predict_declipping(
        self,
        y_clipped,
        clip_value
        ):
        self.clip_value=clip_value

        degradation=lambda x: self.apply_clip(x)

        if self.rid:
            res, denoised, t=self.predict_conditional(y_clipped, degradation)
            return res, denoised, t
        else: 
            res=self.predict_conditional(y_clipped, degradation)
            return res

class SamplerInpainting(Sampler):

    def __init__(self, model, diff_params, args, xi=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, xi, order, data_consistency, rid)

    def apply_mask(self, x):
        return self.mask*x

    def predict_inpainting(
        self,
        y_masked,
        mask
        ):
        self.mask=mask.to(y_masked.device)

        degradation=lambda x: self.apply_mask(x)

        if self.rid: raise NotImplementedError
        res=self.predict_conditional(y_masked, degradation)
        return res

class SamplerBWE(Sampler):

    def __init__(self, model, diff_params, args, xi=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, xi, order, data_consistency, rid)

    def apply_FIR_filter(self,y):
        y=y.unsqueeze(1)

        #apply the filter with a convolution (it is an FIR)
        y_lpf=torch.nn.functional.conv1d(y,self.filt,padding="same")
        y_lpf=y_lpf.squeeze(1) 

        return y_lpf
    def apply_IIR_filter(self,y):
        y_lpf=torchaudio.functional.lfilter(y, self.a,self.b, clamp=False)
        return y_lpf
    def apply_biquad(self,y):
        y_lpf=torchaudio.functional.biquad(y, self.b0, self.b1, self.b2, self.a0, self.a1, self.a2)
        return y_lpf
    def decimate(self,x):
        return x[...,0:-1:self.factor]

    def resample(self,x):
        N=100
        return torchaudio.functional.resample(x,orig_freq=int(N*self.factor), new_freq=N)

    def predict_bwe(
        self,
        ylpf,  #observations (lowpssed signal) Tensor with shape (L,)
        filt, #filter Tensor with shape ??
        filt_type
        ):

        #define the degradation model as a lambda
        if filt_type=="firwin":
            self.filt=filt.to(ylpf.device)
            degradation=lambda x: self.apply_FIR_filter(x)
        elif filt_type=="firwin_hpf":
            self.filt=filt.to(ylpf.device)
            degradation=lambda x: self.apply_FIR_filter(x)
        elif filt_type=="cheby1":
            b,a=filt
            self.a=torch.Tensor(a).to(ylpf.device)
            self.b=torch.Tensor(b).to(ylpf.device)
            degradation=lambda x: self.apply_IIR_filter(x)
        elif filt_type=="biquad":
            b0, b1, b2, a0, a1, a2=filt
            self.b0=torch.Tensor(b0).to(ylpf.device)
            self.b1=torch.Tensor(b1).to(ylpf.device)
            self.b2=torch.Tensor(b2).to(ylpf.device)
            self.a0=torch.Tensor(a0).to(ylpf.device)
            self.a1=torch.Tensor(a1).to(ylpf.device)
            self.a2=torch.Tensor(a2).to(ylpf.device)
            degradation=lambda x: self.apply_biquad(x)
        elif filt_type=="resample":
            self.factor =filt
            degradation= lambda x: self.resample(x)
            return self.predict_resample(ylpf,(ylpf.shape[0], self.args.audio_len), degradation)
        elif filt_type=="decimate":
            self.factor =filt
            degradation= lambda x: self.decimate(x)
            return self.predict_resample(ylpf,(ylpf.shape[0], self.args.audio_len), degradation)
           
        else:
           raise NotImplementedError

        return self.predict_conditional(ylpf, degradation)
        
                

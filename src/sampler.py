from tqdm import tqdm
import torch
import inverse_problems
from decSTN_pytorch import decSTNsingle
import scipy.signal
import numpy as np

class SamplerPhaseRetrieval(Sampler):

    def __init__(self, model, diff_params, args, alpha=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, alpha, order, data_consistency, rid)
        assert data_consistency==False
        assert alpha>0

    def apply_mask(self, x):
        return self.mask*x

    def apply_stft(self,x):
        win_size=self.stft_args["win_size"]
        hop_size=self.stft_args["hop_size"]
        window=torch.hamming_window(window_length=win_size).to(x.device)
        x2=torch.cat((x, torch.zeros(x.shape[0],win_size ).to(x.device)),-1)
        X=torch.stft(x2, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
        Y_hat=torch.sqrt(X[...,0]**2 + X[...,1]**2)
        return Y_hat

    def predict_phase(
        spec,
        stft_args
        ):
        self.stft_args=stft_args

        degradation=lambda x: self.apply_stft(x)

        if self.rid: raise NotImplementedError
        res=self.predict(spec, degradation)
        return res
class SamplerCompSens(Sampler):

    def __init__(self, model, diff_params, args, alpha=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, alpha, order, data_consistency, rid)
        assert data_consistency==False
        assert alpha>0

    def apply_mask(self, x):
        return self.mask*x

    def predict_compsens(
        y_masked,
        mask
        ):

        self.mask=mask.to(y_masked.device)

        degradation=lambda x: self.apply_mask(x)

        if self.rid: raise NotImplementedError
        res=self.predict(y_masked, degradation)
        return res

class SamplerDeclipping(Sampler):

    def __init__(self, model, diff_params, args, alpha=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, alpha, order, data_consistency, rid)
        assert data_consistency==False
        assert alpha>0

    def apply_clip(self,x):
        x_hat=torch.clip(x,min=-self.clip_value, max=self.clip_value)
        return x_hat

    def predict_declipping(
        y_clipped,
        clip_value
        ):
        self.clip_value=clip_value

        degradation=lambda x: self.apply_clip(x)

        if self.rid: raise NotImplementedError
        res=self.predict(y_clipped, degradation)
        return res

class SamplerInpainting(Sampler):

    def __init__(self, model, diff_params, args, alpha=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, alpha, order, data_consistency, rid)

    def apply_mask(self, x):
        return self.mask*x

    def predict_inpaintinh(
        y_masked,
        mask
        ):
        self.mask=mask.to(y_masked.device)

        degradation=lambda x: self.apply_mask(x)

        if self.rid: raise NotImplementedError
        res=self.predict(y_masked, degradation)
        return res

class SamplerBWE(Sampler):

    def __init__(self, model, diff_params, args, alpha=0, order=2, data_consistency=False, rid=False):
        super().__init__(model, diff_params, args, alpha, order, data_consistency, rid)

    def apply_FIR_filter(self,y):
        y=y.unsqueeze(1)

        #apply the filter with a convolution (it is an FIR)
        y_lpf=torch.nn.functional.conv1d(y,self.filt,padding="same")
        y_lpf=y_lpf.squeeze(1) 

        return y_lpf

    def predict_bwe(
        ylpf,  #observations (lowpssed signal) Tensor with shape (L,)
        filt, #filter Tensor with shape ??
        ):

        #define the degradation model as a lambda
        self.filt=filt.to(ylpf.device)
        degradation=lambda x: self.apply_FIR_filter(x)

        if self.rid: raise NotImplementedError
        res=self.predict(ylpf, degradation)
        return res
        
class Sampler():

    def __init__(self, model, diff_params, args, alpha=0, order=2, data_consistency=False, rid=False):

        self.model = model
        self.diff_params = diff_params
        self.alpha=alpha #hyperparameter for the reconstruction guidance
        self.order=order
        self.no_replace=not(data_consistency) #use reconstruction gudance without replacement
        self.args=args
        self.nb_steps=self.args.inference.T

        self.treshold_on_grads=args.inference.max_thresh_grads
        self.rid=rid


    def data_consistency_step(self, x_hat, y, degradation):
        #get reconstruction estimate
        den_rec= degradation(x_hat)     
        #apply replacment (valid for linear degradations)
        return y+x_hat-den_rec 

    def get_score_rec_guidance(self, x, y, t_i, degradation)

        x.requires_grad_()
        x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
        den_rec= degradation(x_hat) 

        norm=torch.linalg.norm(y-den_rec, ord=2)
        
        rec_grads=torch.autograd.grad(outputs=norm,
                                      inputs=x)

        rec_grads=rec_grads[0]
        
        normguide=torch.linalg.norm(rec_grads)/self.args.audio_len**0.5
        
        #normalize scaling
        s=self.alpha/(normguide*t_i+1e-6)
        
        #optionally apply a treshold to the gradients
        if self.treshold_on_grads>0:
            #pply tresholding to the gradients. It is a dirty trick but helps avoiding bad artifacts 
            rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        

        score=(x_hat.detach()-x)/t_i**2

        #apply scaled guidance to the score
        score=score-s*rec_grads

        return score

    def get_score(self,x. y, t_i, degradation):
        if y==None:
            assert degragation==None:
            #unconditional sampling
            x_hat=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
            score=(x_hat-x)/t_i**2
            return score
        else:
            if self.alpha>0:
                #apply rec. guidance
                score=self.get_score_rec_guidance(x, y, t_i, degradation)
    
                #optionally apply replacement or consistency step
                if not(self.no_replace):
                    #convert score to denoised estimate using Tweedie's formula
                    x_hat=score*t_i**2+x
    
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

            score=self.get_score(x_hat, None, t_hat, None)    

            #d=-t_hat*((denoised-x_hat)/t_hat**2)
            d=-t_hat*score
            
            #apply second order correction
            h=t[i+1]-t_hat

            if self.rid: data_denoised[i]=score*t_hat**2+x_hat

            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                score=self.get_score(x_prime, None, t_prime, None)

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d
            
        if self.rid:
            return x.detach(), data_denoised.detach(), t.detach()
        else:
            return x.detach()

    def predict(
        self,
        y,  #observations (lowpssed signal) Tensor with shape ??
        degradation, #lambda function
    ):

        device=y.device
        shape=y.shape

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

            score=self.get_score(x_hat, y, t_hat, degradation)    

            #d=-t_hat*((denoised-x_hat)/t_hat**2)
            d=-t_hat*score
            
            #apply second order correction
            h=t[i+1]-t_hat

            if self.rid: data_denoised[i]=score*t_hat**2+x_hat

            if t[i+1]!=0 and self.order==2:  #always except last step
                #second order correction2
                #h=t[i+1]-t_hat
                t_prime=t[i+1]
                x_prime=x_hat+h*d
                score=self.get_score(x_prime, y, t_prime, degradation)

                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d
            
        if self.rid:
            return x.detach(), data_denoised.detach(), t.detach()
        else:
            return x.detach()

                

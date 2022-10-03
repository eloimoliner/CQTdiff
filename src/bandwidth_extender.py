from tqdm import tqdm
import torch
import inverse_problems
from decSTN_pytorch import decSTNsingle
import scipy.signal
import numpy as np
def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

class BandwidthExtender():
    def __init__(self, args, model, diff_params, device, model_s=None,  diff_params_s=None):
        self.args=args

        self.model=model
        self.diff_params=diff_params

        self.device=device
        self.seg_len=self.args.audio_len            

        self.sampler=Sampler_BWE(self.model, self.diff_params, self.args, alpha=self.args.inference.alpha, order=2, no_replace=self.args.inference.no_replace)
        self.discarded_samples=self.args.inference.long_sampling.discard_samples

        self.filter_final=self.get_FIR_lowpass(100,10000,1,self.args.sample_rate)
        self.filter_final=self.filter_final.to(self.device)

        
    def get_FIR_lowpass(self,order,fc, beta, sr):
    
        B=scipy.signal.firwin(numtaps=order,cutoff=fc, width=beta,window="kaiser", fs=sr)
        B=torch.FloatTensor(B)
        B=B.unsqueeze(0)
        B=B.unsqueeze(0)
        return B

    def process_OLA(
        self,
        y_lpf, #lpf observations shape: (1, T)
        filt #filter shape: (1, N)
    ):
        #y_lpf=y_lpf[..., 0:1000000]#rjust for testing remove this line!!!

        filt=filt.to(self.device)
        #winsize=4092 #samples (46 ms)
        #window=torch.Tensor(np.hanning(2*winsize))
        #window_right=window[winsize::].to(self.device)

        length_data=y_lpf.shape[-1]
        overlapsize=1024 #samples (46 ms)
        window=torch.Tensor(np.hanning(2*overlapsize))
        window_right=window[overlapsize::].to(self.device)
        window_left=window[0:overlapsize].to(self.device)
        audio_finished=False
        pointer=0
        bwe_data=torch.zeros((length_data,)).to(self.device)
        numchunks=int(np.ceil(length_data/self.args.audio_len))
        segment_size=self.args.audio_len

        for i in tqdm(range(numchunks)):
            if pointer+segment_size<length_data:
                segment=y_lpf[...,pointer:pointer+segment_size]
                segment=segment.to(self.device)

                x_pred=self.process_segment(segment, filt)
                x_pred=self.apply_lowpass(x_pred, self.filter_final) #apply lpf to get rid of the artifacts near the nyquist freq, caused because of padding at the border
                x_pred=x_pred.squeeze(0)

                print(x_pred.shape)

                if pointer==0:
                    print(x_pred.shape, window_right)
                    x_pred=torch.cat( (x_pred[0:int(segment_size-overlapsize)], x_pred[int(segment_size-overlapsize):segment_size]*window_right) ,0)
                else:
                    x_pred=torch.cat((x_pred[0:int(overlapsize)]* window_left, x_pred[int(overlapsize):int(segment_size-overlapsize)], x_pred[int(segment_size-overlapsize):int(segment_size)]*window_right), 0)
                    
                bwe_data[pointer:pointer+segment_size]=bwe_data[pointer:pointer+segment_size]+x_pred

                pointer=pointer+segment_size-overlapsize

            else:
                segment=y_lpf[...,pointer::]
                
                lensegment=segment.shape[-1]
                print(segment.shape)
                segment=torch.cat((segment, torch.zeros(1,(int(segment_size-lensegment)),)), -1)
                segment=segment.to(self.device)
                print(segment.shape)
                
                audio_finished=True
                #dostft
                #segment=torch.unsqueeze(segment,0)#is it necessary?

                x_pred=self.process_segment(segment, filt)
                x_pred=self.apply_lowpass(x_pred, self.filter_final) #apply lpf to get rid of the artifacts near the nyquist freq, caused because of padding at the border
                x_pred=x_pred.squeeze(0)
                   
                if pointer!=0:
                    x_pred=torch.cat((x_pred[0:int(overlapsize)]*window_left, x_pred[int(overlapsize):int(segment_size)]),0)
                    x_pred=torch.cat((x_pred[0:int(overlapsize)]*window_left, x_pred[int(overlapsize):int(segment_size)]),0)
                bwe_data[pointer::]=bwe_data[pointer::]+x_pred[0:lensegment]

        return bwe_data

                
    def process(
        self,
        y, #lpf observations shape: (1, T)
        filt #filter shape: (1, N)
    ):
        #y=y[..., 0:200000]#rjust for testing remove this line!!!

        filt=filt.to(self.device)
        #winsize=4092 #samples (46 ms)
        #window=torch.Tensor(np.hanning(2*winsize))
        #window_right=window[winsize::].to(self.device)

        ov=int(self.args.inference.long_sampling.overlap)
        
        if ov>0:
            #assert (ov) > (2*winsize)
            #we need to iterate over the long file (with overlap) in an autoregressive way
            Mt=torch.zeros((1,self.seg_len)).to(self.device)
            #Mt[...,0:int(ov*self.seg_len)]=1
            Mt[...,0:(ov)]=1
            #Mt[...,(ov-winsize):(ov)]=window_right
             
        else:
            Mt=None

        N=y.shape[-1]
        #step=int((1-ov)*self.seg_len)
        step=int(self.seg_len-ov)
        yt=None
        x_hat=torch.zeros((1,N+self.seg_len)) #la guarra, allocate output vector.

        if self.args.architecture=="unet_CQT_nobias":
            self.sampler.model.setup_CQT_len(self.seg_len)
        
        for pointer in tqdm(range(0,N,step-self.discarded_samples)):
            #print("processed ",pointer," from ",N)

            if pointer+self.seg_len <N:
                #normal scenario
                segment=y[...,pointer:pointer+self.seg_len]
                segl=self.seg_len
            else:
                segment=y[...,pointer::]
                
                if self.args.architecture=="unet_CQT_nobias":
                    segment=torch.cat((segment, torch.zeros((1,self.discarded_samples))),-1)
                    segl=segment.shape[-1]
                    self.sampler.model.setup_CQT_len(segl)
                else:
                    segment=torch.cat((segment, torch.zeros((1,self.seg_len-segment.shape[-1]))),-1)
                    segl=segment.shape[-1]

                #segl=next_power_of_2(segl_unused)
                #extra_zeroes=segl-segl_unused
                #segment=torch.cat((segment, torch.zeros((1,extra_zeroes))),-1)
                
                
                
            segment=segment.to(self.device)

            if pointer>0:
                if Mt !=None:
                    yt=x_hat[...,pointer:pointer+segl].to(self.device) #simple but I think is ok... will it probably go out of bounds?
                    #print(yt.shape, segment.shape)
                    assert yt.shape ==segment.shape
                    #print(Mt[...,(ov-self.discarded_samples-2)::], yt[...,(ov-self.discarded_samples-2)::])
                    #print(Mt[...,(ov-self.discarded_samples)::], yt[...,(ov-self.discarded_samples)::])
                    #assert torch.sum((1-Mt[...,(ov-self.discarded_samples):segl])*yt) == 0, "mask is wrong!" +str(torch.sum((1-Mt[...,(ov-self.discarded_samples):segl])*yt))
                    #yt=yt[...,0:segl]
                    #print(segl)
                    #print(segment.shape, yt.shape)
                    seg_hat=self.process_segment(segment, filt, yt=yt, Mt=Mt[...,0:segl])
                else:
                    print(segment.shape)    
                    seg_hat=self.process_segment(segment, filt)
            else:
                #the first iteration has no time conditioning
                seg_hat=self.process_segment(segment, filt)

            seg_hat=self.apply_lowpass(seg_hat, self.filter_final) #apply lpf to get rid of the artifacts near the nyquist freq, caused because of padding at the border

            if self.discarded_samples >0:
                seg_hat=seg_hat[...,0:-self.discarded_samples]

            x_hat[...,pointer:pointer+segl-self.discarded_samples]=seg_hat.cpu() #I will probably need to discard the last samples to avoid clicking problems


        return x_hat[...,0:N]   
        
    def apply_lowpass(self,y,filter):
    
        #ii=1
        B=filter
        y=y.unsqueeze(1)
        #weight=torch.nn.Parameter(B)
        
        y_lpf=torch.nn.functional.conv1d(y,B,padding="same")
        y_lpf=y_lpf.squeeze(1) #some redundancy here, but its ok
        #y_lpf=y
        return y_lpf

    def process_segment(self, y, filt, yt=None, Mt=None):
        

        
        res=self.sampler.predict_bwe(y, filt, yt=yt, Mt=Mt)

        return res 
        

class Sampler_BWE():

    def __init__(self, model, diff_params, args, alpha=0, order=2, no_replace=True, rid=False):
        self.model = model
        self.diff_params = diff_params
        self.alpha=alpha #hyperparameter for the reconstruction guidance
        self.order=order
        self.no_replace=no_replace #use reconstruction gudance without replacement
        self.args=args
        self.nb_steps=self.args.inference.T
        #self.alpha=1 #hyparparameter, 1 equals Heun 2nd order, differnt values can be chosen empirically. They say 1.1 works better

        ##self.use_treshold_on_grads=args.inference.max_thresh_grads!=0
        self.treshold_on_grads=args.inference.max_thresh_grads
        self.use_dynamic_thresh=args.inference.use_dynamic_thresh
        self.dyn_thresh_percentile=args.inference.dynamic_thresh_percentile
        self.rid=rid

    def apply_filter(self,y,filter):
        #ii=1
        y=y.unsqueeze(1)
        #weight=torch.nn.Parameter(B)
        y_lpf=torch.nn.functional.conv1d(y,filter,padding="same")
        y_lpf=y_lpf.squeeze(1) #some redundancy here, but its ok
        #y_lpf=y
        return y_lpf

    def apply_smask(self,x, M):
        
        L=x.shape[-1]
        #careful, these parameters should match how the mask is generated
        nWin1=self.args.STN.nwin
        nHop1 = nWin1*7//8
        win1 = torch.hann_window(nWin1, periodic=True).to(x.device)
        #first do an STFT
        X = torch.stft(x, nWin1, hop_length=nWin1-nHop1, win_length=nWin1, window=win1,center=True, onesided=True, return_complex=True, normalized=False)
        #mask in the STFT domain and inverse-transform
        out = torch.istft(M*X, nWin1, hop_length=nWin1-nHop1, win_length=nWin1, window=win1, center=True, onesided=True, normalized=False)

        return out

    def apply_tmask(self,x,M):
        #will the shapes be ok??
        return M*x

    def denoising_step(
        self,
        x, #Noisy input
        y, #observations, lpfed signal
        t_i, #timestep
        filt, #lowpass filter
        Mt=None #temporal mask
    ):
        #do the basic denoising
        with torch.no_grad():
            denoised=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                
            den_filtered= self.apply_filter(denoised,filt) 
            den_rec=den_filtered
           
            if Mt != None:
                den_t_masked=self.apply_tmask(denoised, Mt)
                den_rec+= den_t_masked - self.apply_tmask(den_rec, Mt)

            denoised2=y+denoised-den_rec 

        return denoised2 

    def denoising_step_rg(
        self,
        x, #Noisy input
        y, #all of the observations
        t_i, #timestep
        filt, #lpf
        Mt=None #temporal mask
    ):
        #do the denoising step with reconstruction guidance
        x.requires_grad_()
        denoised=self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
        
        #apply rec. guidance
        #check if Mt or S are given to know what to do
        
        den_filtered= self.apply_filter(denoised,filt) 
        den_rec=den_filtered

        if Mt != None:
            den_t_masked=self.apply_tmask(denoised, Mt)
            den_rec+= den_t_masked - self.apply_tmask(den_rec, Mt)
        

        #norm=torch.sum((y-den_rec)**2)
        norm=torch.linalg.norm(y-den_rec, ord=2)
    
        rec_grads=torch.autograd.grad(outputs=norm,
                                      inputs=x)
        #print(rec_grads)
        x.detach_()
        rec_grads=rec_grads[0]

        score=(denoised.detach()-x)/t_i**2

        normscore=torch.linalg.norm(score)
        #normguide=torch.linalg.norm(rec_grads)
        normguide=torch.linalg.norm(rec_grads)/self.args.audio_len**0.5
        #s=self.alpha*normscore/(normguide+1e-5)
        s=self.alpha/(normguide*t_i+1e-6)
        #print(normscore, normguide, s)
        #s=self.alph

        #if self.use_dynamic_thresh:
        #    #assert self.treshold_on_grads>0
                
            
        #    tresh = torch.quantile(
        #    rec_grads.abs(),
        #    self.dyn_thresh_percentile,
        #    dim = -1
        #    )
        #    print("tresh",tresh, "max_grad",torch.max(torch.abs(rec_grads)))
        
        #    rec_grads=torch.clip(rec_grads,min=-tresh, max=tresh) 

            #rec_grads=torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)

        rec_grads=s*rec_grads 

        return score.detach(), rec_grads




    def predict_bwe(
        self,
        ylpf,  #observations (lowpssed signal) Tensor with shape ??
        filt, #filter Tensor with shape ??
        yt=None, #obervations from previous segment
        Mt=None #temporal mask for yt
    ):
        #predict a sample conditioned from some observations 

        #if an extra conditioning observation is given, assert the mask or filter is alo given
        assert (yt is None) == (Mt is None)
        if Mt!=None:
            raise NotImplementedError

        device=ylpf.device
        shape=ylpf.shape

        if self.rid:
            data_denoised=torch.zeros((self.nb_steps,shape[0], shape[1]))

        #get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)
        #sample from gaussian distribution with sigma_max variance
        x = self.diff_params.sample_prior(shape,t[0]).to(device)

        #parameter for langevin stochasticity, if Schurn is 0, gamma will be 0 to, so the sampler will be deterministic
        gamma=self.diff_params.get_gamma(t).to(device)

        #put the conditions together
        y=ylpf #y vector will contain all the conditioning signals 

        if yt!=None:
            #add conciously the temporal conditioning
            y+= yt- self.apply_tmask(y, Mt)

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

            if self.alpha>0:
                #denoise with reconstruction guidance
                #S or Mt will probably be None, this function will take care of that
                score, guide=self.denoising_step_rg(x_hat,y,t_hat, filt, Mt)
                score=score-guide

                #replacement or consistency
                if not(self.no_replace):
                    den=score*t_hat**2+x_hat
                    den2=y+den-self.apply_filter(den,filt)
                    score=(den2-x_hat)/t_hat**2

            else:
                #denoised with replacement method
                #S or Mt will probably be None, this function will take care of that
                denoised=self.denoising_step(x_hat,y,t_hat,filt, Mt)
                score=(denoised-x_hat)/t_hat**2

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

                if self.alpha>0:
                    #denoise with reconstruction guidance
                    #S or Mt will probably be None, this function will take care of that
                    #denoised=self.denoising_step_rg(x_prime,y,t_prime, filt, Mt)
                    score, guide=self.denoising_step_rg(x_prime,y,t_prime, filt, Mt)
                    score=score-guide
                    #replacement or consistency
                    if not(self.no_replace):
                        den=score*t_prime**2+x_prime
                        den2=y+den-self.apply_filter(den,filt)
                        score=(den2-x_prime)/t_prime**2
                else:
                    #denoised with replacement method
                    #S or Mt will probably be None, this function will take care of that
                    denoised=self.denoising_step(x_prime,y,t_prime,filt, Mt)
                    score=(denoised-x_prime)/t_prime**2

                #d_prime=-t_prime*((denoised-x_prime)/t_prime**2)
                d_prime=-t_prime*score

                x=(x_hat+h*((1/2)*d +(1/2)*d_prime))

            elif t[i+1]==0 or self.order==1: #first condition  is to avoid dividing by 0
                #first order Euler step
                x=x_hat+h*d
            
        if self.rid:
            return x.detach(), data_denoised.detach(), t.detach()
        else:
            return x.detach()

                

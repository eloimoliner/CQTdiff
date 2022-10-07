import torch 
import time
import numpy as np
import cv2
import torchaudio
from collections import OrderedDict
from nsgt.nsgt  import NSGT
from nsgt.nsgt  import LogScale 

def load_denoiser_model(device,args):

    denoiser_model = unet.MultiStage_denoise(unet_args=args.unet)
    denoiser_model.to(device)
    checkpoint_denoiser=args.denoiser.checkpoint
    try:
        denoiser_model.load_state_dict(torch.load(checkpoint_denoiser, map_location=device))
    except:
        print("Force loading!!!")
        check_point = torch.load(checkpoint_denoiser, map_location=device)
        check_point.key()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        denoiser_model.load_state_dict(new_state_dict)

    return denoiser_model

def add_high_freqs(data, f_dim=1025):
    dims=list(data.size())
    dims[3]=1025
    b=torch.zeros(dims) 
    b=b.to(data.device)
    data=torch.cat((data[:,:,:,0:f_dim],b[:,:,:,0:1025-f_dim]),3)
    return data

def do_istft(data, win_size=2048, hop_size=512, device="cuda"):
    window=torch.hamming_window(window_length=win_size) #this is slow! consider optimizing
    window=window.to(device)
    data=data.permute(0,3,2,1)
    pred_time=torch.istft(data, win_size, hop_length=hop_size,  window=window, center=False, return_complex=False)
    return pred_time

def remove_high_freqs(noisy, clean=None, f_dim=1025):
    noisy=noisy[:,:,:,0:f_dim] 
    if clean!=None:
        clean=clean[:,:,:,0:f_dim] 
        return noisy, clean
    else:
        return noisy

def apply_hard_filter(X,cutoff=5000, fs=44100):
    cut=int(cutoff/(fs/2) *X.shape[-1])
    zeros=torch.zeros_like(X)
    X[:,:,:,cut+1::]=zeros[:,:,:,cut+1::]
    return X
       


class CQT_cpx():
    def __init__(self,fmin, fbins, fs=44100, audio_len=44100, device="cpu", split_0_nyq=False):
        fmax=fs/2
        Ls=audio_len
        scl = LogScale(fmin, fmax, fbins)
        self.cq = NSGT(scl, fs, Ls,
                 real=True,
                 matrixform=True, 
                 multichannel=False,
                 device=device
                 )
        #self.cq=self.cq.to(device)
        self.split_0_nyq=split_0_nyq

    def fwd(self,x):
        #print(x.shape)
        c= self.cq.forward(x)
        #print(c.shape)
        c=c.squeeze(0)
        c= torch.view_as_real(c)
        c=c.permute(0,2,1,3)
        if self.split_0_nyq:
            c0=c[:,:,0,:]
            cnyq=c[:,:,-1,:]
            cfreqs=c[:,:,1:-1,:]
            return cfreqs, c0, cnyq
        else:
            return c

    def bwd(self,cfreqs,c0=None, cnyq=None):

        if self.split_0_nyq:
            c=torch.cat((c0.unsqueeze(2),cfreqs,cnyq.unsqueeze(2)),dim=2)
        else:
            c=cfreqs
        
        c=c.permute(0,2,1,3)
        c=c.unsqueeze(0)
        c=c.contiguous()
        c=torch.view_as_complex(c)
        xrec= self.cq.backward(c)
        #print(xrec.shape)
        return xrec

class CQT():
    def __init__(self,fmin, fbins, fs=44100, audio_len=44100, device="cpu", split_0_nyq=False):
        fmax=fs/2
        Ls=audio_len
        scl = LogScale(fmin, fmax, fbins)
        self.cq = NSGT(scl, fs, Ls,
                 real=True,
                 matrixform=True, 
                 multichannel=False,
                 device=device
                 )
        #self.cq=self.cq.to(device)
        self.split_0_nyq=split_0_nyq

    def fwd(self,x):
        c= self.cq.forward(x)
        c=c.squeeze(0)
        c= torch.view_as_real(c)
        c=c.permute(0,3,2,1)
        if self.split_0_nyq:
            c0=c[:,:,:,0]
            cnyq=c[:,:,:,-1]
            cfreqs=c[:,:,:,1:-1]
            return cfreqs, c0, cnyq
        else:
            return c

    def bwd(self,cfreqs,c0=None, cnyq=None):

        if self.split_0_nyq:
            c=torch.cat((c0.unsqueeze(3),cfreqs,cnyq.unsqueeze(3)),dim=3)
        else:
            c=cfreqs
        
        c=c.permute(0,3,2,1)
        c=c.unsqueeze(0)
        c=c.contiguous()
        c=torch.view_as_complex(c)
        xrec= self.cq.backward(c)
        return xrec


def do_stft_NLD(noisy, win_size=2048, hop_size=512, device="cpu"):
    
    #window_fn = tf.signal.hamming_window

    #win_size=args.stft.win_size
    #hop_size=args.stft.hop_size
    window=torch.hamming_window(window_length=win_size)
    window=window.to(noisy.device)
    noisy=torch.cat((noisy, torch.zeros(noisy.shape[0],noisy.shape[1],win_size).to(noisy.device)),2) 
    noisy_rs=noisy.view(noisy.shape[0]*noisy.shape[1], -1)

    stft_signal_noisy=torch.stft(noisy_rs, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)

    stft_signal_noisy=stft_signal_noisy.view(noisy.shape[0], noisy.shape[1], stft_signal_noisy.shape[1], stft_signal_noisy.shape[2],stft_signal_noisy.shape[3])
    stft_signal_noisy=stft_signal_noisy.permute(0,1, 4,3,2)
    stft_signal_noisy=stft_signal_noisy.reshape(stft_signal_noisy.shape[0], stft_signal_noisy.shape[1]*stft_signal_noisy.shape[2], stft_signal_noisy.shape[3], stft_signal_noisy.shape[4])
    #stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
    #stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)
    

    return stft_signal_noisy

def do_stft(noisy, clean=None, win_size=2048, hop_size=512, device="cpu", DC=True):
    
    #window_fn = tf.signal.hamming_window

    #win_size=args.stft.win_size
    #hop_size=args.stft.hop_size
    window=torch.hamming_window(window_length=win_size)
    window=window.to(noisy.device)
    noisy=torch.cat((noisy, torch.zeros(noisy.shape[0],win_size).to(noisy.device)),1)
    stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
    stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)
    #stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
    #stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)
    
    if clean!=None:

       # stft_signal_clean=tf.signal.stft(clean,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
        clean=torch.cat((clean, torch.zeros(clean.shape[0],win_size).to(device)),1)
        stft_signal_clean=torch.stft(clean, win_size, hop_length=hop_size,window=window, center=False,return_complex=False)
        stft_signal_clean=stft_signal_clean.permute(0,3,2,1)
        #stft_clean_stacked=tf.stack( values=[tf.math.real(stft_signal_clean), tf.math.imag(stft_signal_clean)], axis=-1)


        if DC:
            return stft_signal_noisy, stft_signal_clean
        else:
            return stft_signal_noisy[...,1:], stft_signal_clean[...,1:]
    else:

        if DC:
            return stft_signal_noisy
        else:
            return stft_signal_noisy[...,1:]

#def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
#    if torch.min(y) < -1.:
#        print('min value is ', torch.min(y))
#    if torch.max(y) > 1.:
#        print('max value is ', torch.max(y))
#
#    global mel_basis, hann_window
#    if fmax not in mel_basis:
#        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
#        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
#        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
#
#    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
#    y = y.squeeze(1)
#
#    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
#                      center=center, pad_mode='reflect', normalized=False, onesided=True)
#
#    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
#
#    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
#    spec = spectral_normalize_torch(spec)
#
#    return spec 
def computeSNR(ref,x): #check shapes
    # Signal-to-noise ratio for torch tensors
    ref_pow = (ref**2).mean(-1) + np.finfo('float32').eps
    dif_pow = ((x - ref)**2).mean(-1) + np.finfo('float32').eps
    snr_val = 10 * torch.log10(ref_pow / dif_pow)
    snr_val= torch.sum(snr_val,axis=0)
    return snr_val

def computeLSD(y,y_g): #check!!!
    yF=do_stft(y,win_size=2048, hop_size=512)
    ygF=do_stft(y_g, win_size=2048, hop_size=512)
    yF=torch.sqrt(yF[:,0,:,:]**2 +yF[:,1,:,:]**2)
    ygF=torch.sqrt(ygF[:,0,:,:]**2 +ygF[:,1,:,:]**2)
    Sy = torch.log10(torch.abs(yF)**2 + 1e-8)
    Syg = torch.log10(torch.abs(ygF)**2 + 1e-8)
    lsd = torch.sum(torch.mean(torch.sqrt(torch.mean((Sy-Syg)**2 + 1e-8, axis=2)), axis=1), axis=0)

    return lsd

def computeHFLSD(y,y_g, fc,fs): #check!!!
    index=int((fc*2048)/fs)
    yF=do_stft(y,win_size=2048, hop_size=512)
    ygF=do_stft(y_g, win_size=2048, hop_size=512)
    yF=torch.sqrt(yF[:,0,:,index:-1]**2 +yF[:,1,:,index:-1]**2)
    ygF=torch.sqrt(ygF[:,0,:,index:-1]**2 +ygF[:,1,:,index:-1]**2)
    Sy = torch.log10(torch.abs(yF)**2 + 1e-8)
    Syg = torch.log10(torch.abs(ygF)**2 + 1e-8)
    lsd = torch.sum(torch.mean(torch.sqrt(torch.mean((Sy-Syg)**2 + 1e-8, axis=2)), axis=1), axis=0)

    return lsd
def computeLFLSD(y,y_g,fc,fs): #check!!!
    index=int((fc*2048)/fs)
    yF=do_stft(y,win_size=2048, hop_size=512)
    ygF=do_stft(y_g, win_size=2048, hop_size=512)
    yF=torch.sqrt(yF[:,0,:,0:index]**2 +yF[:,1,:,0:index]**2)
    ygF=torch.sqrt(ygF[:,0,:,0:index]**2 +ygF[:,1,:,0:index]**2)
    Sy = torch.log10(torch.abs(yF)**2 + 1e-8)
    Syg = torch.log10(torch.abs(ygF)**2 + 1e-8)
    lsd = torch.sum(torch.mean(torch.sqrt(torch.mean((Sy-Syg)**2 + 1e-8, axis=2)), axis=1), axis=0)

    return lsd
  
#def generate_images_from_spec(spec):
#    spec=spec.permute(2,1,0)
#    spec=spec.unsqueeze(0)
#    cpx=spec.cpu().detach().numpy()
#    spectro=np.clip((np.flipud(np.transpose(10*np.log10(np.sqrt(np.power(cpx[...,0],2)+np.power(cpx[...,1],2)))))+30)/50,0,1)
#    cmap=cv2.COLORMAP_JET
#    spectro = np.array((1-spectro)* 255, dtype = np.uint8)
#    spectro = cv2.applyColorMap(spectro, cmap)
#    return np.flipud(np.fliplr(spectro))
def generate_images(audio):
    audio=audio.cpu()
    audio=audio.unsqueeze(0)
    cpx=do_stft(audio, win_size=1025, hop_size=256)
    cpx=cpx.permute(0,3,2,1)
    cpx=cpx.cpu().detach().numpy()
    spectro=np.clip((np.flipud(np.transpose(10*np.log10(np.sqrt(np.power(cpx[...,0],2)+np.power(cpx[...,1],2)))))+30)/50,0,1)
    cmap=cv2.COLORMAP_JET
    spectro = np.array((1-spectro)* 255, dtype = np.uint8)
    spectro = cv2.applyColorMap(spectro, cmap)
    return np.flipud(np.fliplr(spectro))

def generate_images_Amat(Amat):
    lenamats=len(Amat) 
    for i in range(lenamats):
        att=Amat[i].cpu().detach().numpy()
        att=np.clip((10*np.log10(att+1e-8)+40)/20, 0,1)
        cmap=cv2.COLORMAP_JET
        att = np.array((1-att)* 255, dtype = np.uint8)
        att = cv2.applyColorMap(att, cmap)
        if i==0:
             res=att
        else:
             res=np.concatenate((res,att), axis=0)
    return res
def generate_images_from_spec_mag(CQTmat):
    CQTmat=CQTmat.permute(0,3,2,1)
    
    for i in range(CQTmat.shape[0]):
        mag=CQTmat[i].cpu().detach().numpy()
        mag=mag.squeeze(2)
        mag=np.clip(mag, 0, None)
        spectro=np.clip((np.flipud(np.transpose(10*np.log10(mag+1e-8)))+30)/50,0,1)
        cmap=cv2.COLORMAP_JET
        spectro = np.array((1-spectro)* 255, dtype = np.uint8)
        spectro = cv2.applyColorMap(spectro, cmap)
        o= np.flipud(np.fliplr(spectro))
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=0)
    return res
def generate_images_from_cpxspec(CQTmat):
    CQTmat=CQTmat.squeeze(1)
    CQTmat=CQTmat.permute(0,2,1,3)
    
    for i in range(CQTmat.shape[0]):
        cpx=CQTmat[i].cpu().detach().numpy()
        spectro=np.clip((np.flipud(np.transpose(10*np.log10(np.sqrt(np.power(cpx[...,0],2)+np.power(cpx[...,1],2)))))+30)/50,0,1)
        cmap=cv2.COLORMAP_JET
        spectro = np.array((1-spectro)* 255, dtype = np.uint8)
        spectro = cv2.applyColorMap(spectro, cmap)
        #print(spectro.shape)
        o= np.flipud(np.fliplr(spectro))
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=0)
    return res
def generate_images_from_cqt(CQTmat):
    CQTmat=CQTmat.permute(0,3,2,1)
    
    for i in range(CQTmat.shape[0]):
        cpx=CQTmat[i].cpu().detach().numpy()
        spectro=np.clip((np.flipud(np.transpose(10*np.log10(np.sqrt(np.power(cpx[...,0],2)+np.power(cpx[...,1],2)))))+30)/50,0,1)
        cmap=cv2.COLORMAP_JET
        spectro = np.array((1-spectro)* 255, dtype = np.uint8)
        spectro = cv2.applyColorMap(spectro, cmap)
        o= np.flipud(np.fliplr(spectro))
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=0)
    return res
#def generate_images_cqt(audio, CQTransform):
#    audio=audio.unsqueeze(0)
#    cpx=CQTransform.fwd(audio)
#    cpx=cpx.permute(0,3,2,1)
#    cpx=cpx.cpu().detach().numpy()
#    spectro=np.clip((np.flipud(np.transpose(10*np.log10(np.sqrt(np.power(cpx[...,0],2)+np.power(cpx[...,1],2)))))+30)/50,0,1)
#    cmap=cv2.COLORMAP_JET
#    spectro = np.array((1-spectro)* 255, dtype = np.uint8)
#    spectro = cv2.applyColorMap(spectro, cmap)
#    return np.flipud(np.fliplr(spectro))

def apply_nlds(x):
    #H=[0, 1.4847, 0, -1.4449, 0, 2.4713, 0, -2.4234, 0, 0.9158, 0];  #FEXP!  Analytical and Perceptual Evaluation of Nonlinear Devices for Virtual Bass System, Nay Oo , and Woon-Seng Gan
    #x1=torch.zeros_like(x)
    #for n in range(1,len(H)):
    #    x1=x1+H[n]*x**(n)
    #HWR, RELU is the same

    x1=torch.nn.functional.tanh(x)
    x2=torch.nn.functional.relu(x)
    

    return torch.stack((x1,x2), dim=1)

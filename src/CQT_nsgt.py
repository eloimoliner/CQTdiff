import torch 
from src.nsgt  import NSGT
from src.nsgt  import LogScale 


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



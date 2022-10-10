import scipy.signal
import torch

def get_FIR_lowpass(order,fc, beta, sr):

    B=scipy.signal.firwin(numtaps=order,cutoff=fc, width=beta,window="kaiser", fs=sr)
    B=torch.FloatTensor(B)
    B=B.unsqueeze(0)
    B=B.unsqueeze(0)
    return B

def apply_low_pass(y,filter):

    #ii=2
    B=filter
    y=y.unsqueeze(1)
    #weight=torch.nn.Parameter(B)
    
    y_lpf=torch.nn.functional.conv1d(y,B,padding="same")
    y_lpf=y_lpf.squeeze(1) #some redundancy here, but its ok
    #y_lpf=y
    return y_lpf

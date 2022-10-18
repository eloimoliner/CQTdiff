import scipy.signal
import torch
import torchaudio
import math

def get_FIR_high_pass(order,fc, beta, sr):
    """
        This function designs a  FIR high pass filter using the window method. It uses scipy.signal
        Args:
            order(int): order of the filter
            fc (float): cutoff frequency
            sr (float): sampling rate

        Returns:
            B (Tensor): shape(1,1,order-1) FIR filter coefficients
    """

    B=scipy.signal.firwin(numtaps=order-1,cutoff=fc, width=beta,window="kaiser", fs=sr, pass_zero="highpass")
    B=torch.FloatTensor(B)
    B=B.unsqueeze(0)
    B=B.unsqueeze(0)
    return B
def get_FIR_lowpass(order,fc, beta, sr):
    """
        This function designs a FIR low pass filter using the window method. It uses scipy.signal
        Args:
            order(int): order of the filter
            fc (float): cutoff frequency
            sr (float): sampling rate
        Returns:
            B (Tensor): shape(1,1,order) FIR filter coefficients
    """

    B=scipy.signal.firwin(numtaps=order,cutoff=fc, width=beta,window="kaiser", fs=sr)
    B=torch.FloatTensor(B)
    B=B.unsqueeze(0)
    B=B.unsqueeze(0)
    return B

def apply_low_pass_firwin(y,filter):
    """
        Utility for applying a FIR filter, usinf pytorch conv1d
        Args;
            y (Tensor): shape (B,T) signal to filter
            filter (Tensor): shape (1,1,order) FIR filter coefficients
        Returns:
            y_lpf (Tensor): shape (B,T) filtered signal
    """

    #ii=2
    B=filter.to(y.device)
    y=y.unsqueeze(1)
    #weight=torch.nn.Parameter(B)
    
    y_lpf=torch.nn.functional.conv1d(y,B,padding="same")
    y_lpf=y_lpf.squeeze(1) #some redundancy here, but its ok
    #y_lpf=y
    return y_lpf

def apply_decimate(y,factor):
    """
        Function for applying a naive decimation for downsampling
        Args:
            y (Tensor): shape (B,T)
            factor (int): decimation factor
        Returns
            y (Tensor): shape (B,T//factor)
    """

    factor=factor
    return y[...,0:-1:factor]

def apply_resample(y,factor):
    """
        Applies torch's resmpling function
        Args:
            y (Tensor): shape (B,T)
            factor (float): resampling factor
    """
    N=100 
    return torchaudio.functional.resample(y,orig_freq=int(factor*N),new_freq=N )

def apply_low_pass_biquad(y,filter):
    """
        Applies torchaudio's biquad filter
        Args:
            y (Tensor): shape (B,T)
            filter (tuple): biquad filter coefficients
        Returns:        
            y_lpf (Tensor) : shape (B,T) filtered signal
    """
    b0,b1,b2,a0,a1,a2=filter
    b0=torch.Tensor(b0).to(y.device)
    b1=torch.Tensor(b1).to(y.device)
    b2=torch.Tensor(b2).to(y.device)
    a0=torch.Tensor(a0).to(y.device)
    a1=torch.Tensor(a1).to(y.device)
    a2=torch.Tensor(a2).to(y.device)
    y_lpf=torchaudio.functional.biquad(y, b0,b1,b2,a0,a1,a2)
    return y_lpf
def apply_low_pass_IIR(y,filter):
    b,a=filter
    b=torch.Tensor(b).to(y.device)
    a=torch.Tensor(a).to(y.device)
    y_lpf=torchaudio.functional.lfilter(y, a,b, clamp=False)
    return y_lpf

def apply_low_pass(y,filter, type):
    """
        Meta-function for applying a lowpass filter, maps y to another function depending on the type
        Args:
           y (Tensors): shape (B,T)
           filter (whatever): filter coefficients, or whatever that specifies the filter
           type (string): specifier of the type of filter
        Returns
           y_lpf (Tensor): shape (B,,T) foltered signal
    """

    if type=="firwin":
        return apply_low_pass_firwin(y,filter)
    if type=="firwin_hpf":
        return apply_low_pass_firwin(y,filter)
    elif type=="cheby1":
        return apply_low_pass_IIR(y,filter)
    elif type=="biquad":
        return apply_low_pass_biquad(y,filter)
    elif type=="resample":
        return apply_resample(y,filter)
    elif type=="decimate":
        return apply_decimate(y,filter)

def get_cheby1_ba(order, ripple,hi ):
    """
        Utility for designing a chebyshev type I IIR lowpass filter
        Args:
           order, ripple, hi: (see scipy.signal.cheby1 documentation)
        Returns:
           b,a: filter coefficients
    """
    b,a = scipy.signal.cheby1(order, ripple, hi, btype='lowpass', output='ba')
    return b,a

def design_biquad_lpf(fc, fs, Q):
    """
        utility for designing a biqad lowpass filter
        Args:
            fc (float): cutoff frequency
            fs (float): sampling frequency
            Q (float):  Q-factor
    """
    w0 = 2 * math.pi * fc / fs
    w0 = torch.as_tensor(w0, dtype=torch.float32)
    alpha = torch.sin(w0) / 2 / Q

    b0 = (1 - torch.cos(w0)) / 2
    b1 = 1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return b0, b1, b2, a0, a1, a2


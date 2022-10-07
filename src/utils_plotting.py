import torch 
import time
import numpy as np
import cv2
import torchaudio
import utils
import matplotlib.pyplot as plt
import plotly.express as px
import soundfile as sf
import plotly.graph_objects as go
import pandas as pd
import plotly

def plot_norms(path, normsscores, normsguides, t, name):
    values=t.cpu().numpy()
     
    df=pd.DataFrame.from_dict(
                {"sigma": values[0:-1], "score": normsscores.cpu().numpy(), "guidance": normsguides.cpu().numpy()
                }
                )

    fig= px.line(df, x="sigma", y=["score", "guidance"],log_x=True,  log_y=True, markers=True)


    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)
    return fig
    

def plot_spectral_analysis_sampling( avgspecNF,avgspecDEN, ts, fs=22050, nfft=1024):
    T,F=avgspecNF.shape
    f=torch.arange(0, F)*fs/nfft
    f=f.unsqueeze(1).repeat(1,T).view(-1)
    avgspecNF=avgspecNF.permute(1,0).reshape(-1)
    avgspecDEN=avgspecDEN.permute(1,0).reshape(-1)
    ts=ts.cpu().numpy().tolist()
    ts=513*ts
    #ts=np.repeat(ts,513)
    print(f.shape, avgspecY.shape, avgspecNF.shape, len(ts))
    df=pd.DataFrame.from_dict(
        {"f": f,  "noisy": avgspecNF, "denoised":  avgspecDEN, "sigma":ts
        }
        )

    fig= px.line(df, x="f", y=["noisy", "denoised"],  animation_frame="sigma", log_x=False,log_y=False,  markers=False)
    return fig

def plot_spectral_analysis(avgspecY, avgspecNF,avgspecDEN, ts, fs=22050, nfft=1024):
    T,F=avgspecNF.shape
    f=torch.arange(0, F)*fs/nfft
    f=f.unsqueeze(1).repeat(1,T).view(-1)
    avgspecY=avgspecY.squeeze(0).unsqueeze(1).repeat(1,T).view(-1)
    avgspecNF=avgspecNF.permute(1,0).reshape(-1)
    avgspecDEN=avgspecDEN.permute(1,0).reshape(-1)
    ts=ts.cpu().numpy().tolist()
    ts=513*ts
    #ts=np.repeat(ts,513)
    print(f.shape, avgspecY.shape, avgspecNF.shape, len(ts))
    df=pd.DataFrame.from_dict(
        {"f": f, "y": avgspecY, "noisy": avgspecNF, "denoised":  avgspecDEN, "sigma":ts
        }
        )

    fig= px.line(df, x="f", y=["y", "noisy", "denoised"],  animation_frame="sigma", log_x=False,log_y=False,  markers=False)
    return fig

def plot_loss_by_sigma_test_snr(average_snr, average_snr_out, t):
    #write a fancy plot to log in wandb
    values=np.array(t)
    df=pd.DataFrame.from_dict(
                {"sigma": values, "SNR": average_snr, "SNR_denoised": average_snr_out
                }
                )

    fig= px.line(df, x="sigma", y=["SNR", "SNR_denoised"],log_x=True,  markers=True)
    return fig
    


def plot_loss_by_sigma_test(average_loss, t):
    #write a fancy plot to log in wandb

    return plot_loss_by_sigma(np.array(t), average_loss)


def plot_loss_by_sigma_train(sum_loss_in_bins, num_elems_in_bins, quantized_sigma_values):
    #write a fancy plot to log in wandb
    n_bins=sum_loss_in_bins.size
    valid_bins=num_elems_in_bins>0

    mean_loss_in_bins= (sum_loss_in_bins[valid_bins]/num_elems_in_bins[valid_bins])
    indexes= np.arange(0,n_bins,1)
    indexes=indexes[valid_bins]
    values=quantized_sigma_values[valid_bins]

    return plot_loss_by_sigma(values, mean_loss_in_bins)

# Create and style traces

def plot_loss_by_sigma(values, mean_loss_in_bins):
    df=pd.DataFrame.from_dict(
                {"sigma": values, "loss": mean_loss_in_bins
                }
                )

    fig= px.line(df, x="sigma", y="loss",log_x=True,  markers=True, range_y=[0, 2])
    


    return fig


def plot_melspectrogram(X, refr=1):
    X=X.squeeze(1) #??
    X=X.cpu().numpy()
    if refr==None:
        refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*np.log10(np.abs(X)/refr)

    S_db=np.transpose(S_db, (0,2,1))
    S_db=np.flip(S_db, axis=1)

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow( res,  zmin=-40, zmax=20)

    fig.update_layout(coloraxis_showscale=False)

    return fig

def plot_cpxspectrogram(X):
    X=X.squeeze(1)
    X=X.cpu().numpy()
    #Xre=X[...,0]
    #Xim=X[...,1]
    #X=np.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)
    #if refr==None:
    #    refr=np.max(np.abs(X))+1e-8
     
    #S_db = 10*np.log10(np.abs(X)/refr)

    #S_db=np.transpose(S_db, (0,2,1))
    #S_db=np.flip(S_db, axis=1)

    #for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
    #    o=S_db[i]
    #    if i==0:
    #         res=o
    #    else:
    #         res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow(X, facet_col=3, animation_frame=0)

    fig.update_layout(coloraxis_showscale=False)

    return fig

def plot_mag_spectrogram(X, refr=None, path=None,name="spec"):
    #X=X.squeeze(1)
    X=X.cpu().numpy()
    #X=np.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)
    if refr==None:
        refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*np.log10(np.abs(X)/refr)

    #S_db=np.transpose(S_db, (0,2,1))
    S_db=np.flip(S_db, axis=1)

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow( res,  zmin=-40, zmax=20)

    fig.update_layout(coloraxis_showscale=False)

    path_to_plotly_png = path+"/"+name+".png"
    plotly.io.write_image(fig, path_to_plotly_png)
    
    return fig
def plot_spectrogram(X, refr=None):
    X=X.squeeze(1)
    X=X.cpu().numpy()
    X=np.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)
    if refr==None:
        refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*np.log10(np.abs(X)/refr)

    S_db=np.transpose(S_db, (0,2,1))
    S_db=np.flip(S_db, axis=1)

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=np.concatenate((res,o), axis=1)
      
    fig=px.imshow( res,  zmin=-40, zmax=20)

    fig.update_layout(coloraxis_showscale=False)

    return fig

def write_audio_file(x,sr):
    path="/scratch/work/molinee2/tmp/tmp_audio_example.wav"
    x=x.flatten()
    x=x.unsqueeze(1)
    x=x.cpu().numpy()
    
    sf.write(path,x,sr)
    return path

def plot_cpxCQT_from_raw_audio(x, args, refr=None ):
    #shape of input spectrogram:  (     ,T,F, )
    fmax=args.sample_rate/2
    fmin=fmax/(2**args.cqt.numocts)
    fbins=int(args.cqt.binsoct*args.cqt.numocts) 
    device=x.device
    CQTransform=utils.CQT_cpx(fmin,fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)

    x=x
    X=CQTransform.fwd(x)
    return plot_cpxspectrogram(X)

def plot_CQT_from_raw_audio(x, args, refr=None ):
    #shape of input spectrogram:  (     ,T,F, )
    fmax=args.sample_rate/2
    fmin=fmax/(2**args.cqt.numocts)
    fbins=int(args.cqt.binsoct*args.cqt.numocts) 
    device=x.device
    CQTransform=utils.CQT_cpx(fmin,fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)

    refr=3
    x=x
    X=CQTransform.fwd(x)
    return plot_spectrogram(X, refr)

def get_spectrogram_from_raw_audio(x, stft, refr=1):
    X=utils.do_stft(x, win_size=stft.win_size, hop_size=stft.hop_size)
    X=X.permute(0,2,3,1)

    X=X.squeeze(1)
    #X=X.cpu().numpy()
    X=torch.sqrt(X[:,:,:,0]**2 + X[:,:,:,1]**2)

    #if refr==None:
    #    refr=np.max(np.abs(X))+1e-8
     
    S_db = 10*torch.log10(torch.abs(X)/refr)

    S_db=S_db.permute(0,2,1)
    #np.transpose(S_db, (0,2,1))
    S_db=torch.flip(S_db, [1])

    for i in range(X.shape[0]): #iterate over batch size, shity way of ploting all the batched spectrograms
        o=S_db[i]
        if i==0:
             res=o
        else:
             res=torch.cat((res,o), 1)
    return res

def diffusion_CQT_animation(path, x ,t,  args, refr=1, name="animation_diffusion" ):
    #shape of input spectrograms:  (Nsteps,B,Time,Freq)
    #print(noisy.shape)
    Nsteps=x.shape[0]
    numsteps=10
    tt=torch.linspace(0, Nsteps-1, numsteps)
    i_s=[]
    allX=None
    device=x.device

    fmax=args.sample_rate/2
    fmin=fmax/(2**args.cqt.numocts)
    fbins=int(args.cqt.binsoct*args.cqt.numocts) 
    CQTransform=utils.CQT_cpx(fmin,fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)

    for i in tt:
        i=int(torch.floor(i))
        print(i)
        i_s.append(i)
        xx=x[i]
        print("xx",xx.shape)
        X=CQTransform.fwd(xx)
        print("X",X.shape)
        if allX==None:
             allX=X
        else:
             allX=torch.cat((allX,X), 0)

    print("allX",allX.shape)   #t,B,T,F, cpx
      
    allX=allX.cpu().numpy()
    
    fig=px.imshow(allX, animation_frame=0, facet_col=3) #I need to incorporate t here!!!

    fig.update_layout(coloraxis_showscale=False)
    
    t=t[i_s].cpu().numpy()
       
    assert len(t)==len(fig.frames), print(len(t), len(fig.frames))
    #print(fig)
    for i, f in enumerate(fig.layout.sliders[0].steps):
        #hacky way of changing the axis values
        #f.args[0]=[str(t[i])]
        f.label=str(t[i])

    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)


    return fig
def diffusion_spec_animation(path, x ,t,  stft, refr=1, name="animation_diffusion" ):
    #shape of input spectrograms:  (Nsteps,B,Time,Freq)
    #print(noisy.shape)
    Nsteps=x.shape[0]
    numsteps=10
    tt=torch.linspace(0, Nsteps-1, numsteps)
    i_s=[]
    allX=None
    for i in tt:
        i=int(torch.floor(i))
        print(i)
        i_s.append(i)
        X=get_spectrogram_from_raw_audio(x[i],stft, refr)
        X=X.unsqueeze(0)
        if allX==None:
             allX=X
        else:
             allX=torch.cat((allX,X), 0)

        
        print(allX.shape)
      
    allX=allX.cpu().numpy()
    fig=px.imshow(allX, animation_frame=0,  zmin=-40, zmax=20) #I need to incorporate t here!!!

    fig.update_layout(coloraxis_showscale=False)
    
    t=t[i_s].cpu().numpy()
       
    assert len(t)==len(fig.frames), print(len(t), len(fig.frames))
    #print(fig)
    for i, f in enumerate(fig.layout.sliders[0].steps):
        #hacky way of changing the axis values
        #f.args[0]=[str(t[i])]
        f.label=str(t[i])

    path_to_plotly_html = path+"/"+name+".html"
    
    fig.write_html(path_to_plotly_html, auto_play = False)


    return fig

def plot_spectrogram_from_raw_audio(x, stft, refr=None ):
    #shape of input spectrogram:  (     ,T,F, )
    refr=3
    x=x
    X=utils.do_stft(x, win_size=stft.win_size, hop_size=stft.hop_size)
    X=X.permute(0,2,3,1)
    return plot_spectrogram(X, refr)

def plot_spectrogram_from_cpxspec(X, refr=None):
    #shape of input spectrogram:  (     ,T,F, )
    return plot_spectrogram(X, refr)

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

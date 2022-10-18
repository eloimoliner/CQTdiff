import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math as m
import torch
#import torchaudio
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

import torchaudio
class CombinerUp(nn.Module):

    def __init__(self,mode, Npyr, Nx, bias=True):
        super().__init__()
        self.conv1x1=nn.Conv1d(Nx, Npyr,1, bias=bias)
        self.mode=mode
        #self.GN=nn.GroupNorm(8,Nx)
        torch.nn.init.constant_(self.conv1x1.weight, 0)
    def forward(self,pyr,x):
                
        if self.mode=="sum":
            x=self.conv1x1(x)
            if pyr==None:
                return x
            else:
                
                return (pyr[...,0:x.shape[-1]]+x)/(2**0.5)
                #return (pyr[...,0:x.shape[-1]]+x)/(2**0.5)
        else:
            raise NotImplementedError

class CombinerDown(nn.Module):

    def __init__(self,mode, Nin, Nout, bias=True):
        super().__init__()
        self.conv1x1=nn.Conv1d(Nin, Nout,1, bias=bias)
        self.mode=mode
    def forward(self,pyr,x):
        if self.mode=="sum":
            pyr=self.conv1x1(pyr)
            return (pyr+x)/(2**0.5)
        else:
            raise NotImplementedError

class Upsample(nn.Module):
    def __init__(self,S):
        super().__init__()
        N=2**12
        self.resample=torchaudio.transforms.Resample(N,N*S) #I use 3**12 as an arbitrary number, as we don't care about the sampling frequency of the latents
        #self.resample=nn.Upsample( scale_factor=(1,S))
    def forward(self,x):
        return self.resample(x) 

class Downsample(nn.Module):
    def __init__(self,S):
        super().__init__()
        N=2**12
        self.resample=torchaudio.transforms.Resample(N,N/S) #I use 2**12 as an arbitrary number, as we don't care about the sampling frequency of the latents
        #self.resample=nn.AvgPool2d((1,S+1), stride=(1,S), padding=(0,1))
    def forward(self,x):
        return self.resample(x) 
    

class RFF_MLP_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class FinalBlock(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, 2] 
    Final block. Basiforwardy, a 3x3 conv. layer to map the output features to the output complex spectrogram.

    '''
    def __init__(self, N0):
        super(FinalBlock, self).__init__()
        ksize=(3,3)
        self.conv2=ComplexConv1d(N0,out_channels=1,
                      kernel_size=ksize,
                      stride=1, 
                      padding='same',
                      padding_mode='zeros')


    def forward(self, inputs ):

        pred=self.conv2(inputs)

        return pred




class Film(nn.Module):
    def __init__(self, output_dim, bias=True):
        super().__init__()
        self.bias=bias
        if bias:
            self.output_layer = nn.Linear(512, 2 * output_dim)
        else:
            self.output_layer = nn.Linear(512, 1 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1) #we need a secnond unsqueeze because our data is 2d [B,C,1,1]
        if self.bias:
            gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        else:
            gamma=sigma_encoding
            beta=None

        return gamma, beta

class Gated_residual_layer(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        dilation,
        bias=True
    ):
        super().__init__()
        self.conv= nn.Conv1d(dim,dim,
                  kernel_size=kernel_size,
                  dilation=dilation,
                  stride=1,
                  padding='same',
                  padding_mode='zeros', bias=bias) #freq convolution (dilated) 
        self.act= nn.GELU()
        #self.conv1_1= nn.Conv2d(dim,dim,
        #          kernel_size=1)
        #self.position_gate = nn.Sequential( nn.Linear(64, dim),
        #                                    nn.Sigmoid()) #assuming that 64 is the dimensionality of the RFF freq. positional embeddings
        #self.gn=nn.GroupNorm(8, dim)

    def forward(self, x):
        #gate=self.position_gate(freqembeddings)  #F, N
        #B,N,T,F=x.shape
        #gate = gate.unsqueeze(0).unsqueeze(0) #1,1, F,N
        #gate = gate.permute(0,3,2,1) #1,N,1, F
        #torch.broadcast_to(gate.permute(0,1).unsqueeze(0).unsqueeze(0)
        
        x=(x+self.conv(self.act(x)))/(2**0.5)
        return x
        
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        use_norm=False,
        groups = 8,
        bias=True,
    ):
        super().__init__()

        self.bias=bias
        self.use_norm=use_norm
        self.film=Film(dim, bias=bias)

        self.res_conv = nn.Conv1d(dim, dim_out, 1, padding_mode="zeros", bias=bias) if dim != dim_out else nn.Identity()

        self.H=nn.ModuleList()
        self.num_layers=8

        if self.use_norm:
            self.gnorm=nn.GroupNorm(8,dim)

        self.first_conv=nn.Sequential(nn.GELU(),nn.Conv1d(dim, dim_out,1, bias=bias))

         
        for i in range(self.num_layers):
            self.H.append(Gated_residual_layer(dim_out, 5, 2**i, bias=bias)) #sometimes I changed this 1,5 to 3,5. be careful!!! (in exp 80 as far as I remember)


    def forward(self, x, sigma):
        
        gamma, beta = self.film(sigma)

        if self.use_norm:
            x=self.gnorm(x)

        if self.bias:
            x=x*gamma+beta
        else:
            x=x*gamma #no bias

        y=self.first_conv(x)

        
        for h in self.H:
            y=h(y)

        return (y + self.res_conv(x))/(2**0.5)

class Unet_1d(nn.Module):

    def __init__(self, args, device):
        super(Unet_1d, self).__init__()
        self.args=args
        self.depth=6
        self.embedding = RFF_MLP_Block()
        self.use_norm=args.cqt.use_norm

        #fmax=self.args.sample_rate/2
        #self.fmin=fmax/(2**self.args.cqt.numocts)
        #self.fbins=int(self.args.cqt.binsoct*self.args.cqt.numocts) 
        self.device=device
        #self.CQTransform=utils.CQT_cpx(self.fmin,self.fbins, self.args.sample_rate, self.args.audio_len, device=self.device, split_0_nyq=False)
        Nin=1   

        #self.f_dim=self.args.stft.win_size//2 +1
        #self.f_dim=self.fbins+2
        #N_freq_encoding=32
        #self.freqembeddings=FreqEncodingRFF(self.f_dim, N_freq_encoding).embeddings
        #N_fencoding=32
        #self.freq_encoding=AddFreqEncodingRFF(self.f_dim,N_freq_encoding)
        #Nin=Nin+N_freq_encoding*2 #hardcoded

        #self.use_fencoding=True

        #Encoder
        self.Ns= [64, 64,128,128, 256, 256, 256]
        self.Ss= [2,2,2,2,2,2]
        
        #initial feature extractor
        #ksize=(7,7)

        #self.conv2d_1 = nn.Sequential(nn.Conv2d(Nin,self.Ns[0],
        #              kernel_size=ksize,
        #              padding='same',
        #              padding_mode='reflect'),
        ##              nn.ELU())
                        
        self.init_conv= nn.Conv1d(Nin,self.Ns[0],5, padding="same", padding_mode="zeros", bias=False)
        #self.final_conv= nn.Conv2d(self.Ns[0],2,(3,3), padding="same", padding_mode="reflect")

        #initialize last layer with 0


        self.downs=nn.ModuleList([])
        self.middle=nn.ModuleList([])
        self.ups=nn.ModuleList([])
        
        for i in range(self.depth):
            if i==0:
                dim_in=self.Ns[i]
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i-1]
                dim_out=self.Ns[i]

            if i<(self.depth-1):
                self.downs.append(
                                   nn.ModuleList([
                                            ResnetBlock(dim_in, dim_out, self.use_norm, bias=False),
                                            Downsample( self.Ss[i]),
                                            CombinerDown("sum", 1, dim_out, bias=False)]))

            elif i==(self.depth-1): #no downsampling in the last layer
                self.downs.append(
                                   nn.ModuleList([
                                            ResnetBlock(dim_in, dim_out, self.use_norm, bias=False),
                                            ]))

        self.middle.append(nn.ModuleList([
                        ResnetBlock(self.Ns[self.depth], self.Ns[self.depth], self.use_norm, bias=False)
                        ]))
        for i in range(self.depth-1,-1,-1):

            if i==0:
                dim_in=self.Ns[i]*2
                dim_out=self.Ns[i]
            else:
                dim_in=self.Ns[i]*2
                dim_out=self.Ns[i-1]

            if i>0: 
                self.ups.append(nn.ModuleList(
                                            [
                                            ResnetBlock(dim_in, dim_out, use_norm=self.use_norm, bias=False),
                                            Upsample( self.Ss[i]),
                                            CombinerUp("sum", 1, dim_out, bias=False)]))

            elif i==0: #no downsampling in the last layer
                self.ups.append(
                                   nn.ModuleList([
                                            ResnetBlock(dim_in, dim_out, use_norm=self.use_norm, bias=False),
                                            ]))


        self.cropconcat = CropConcatBlock()



    def setup_CQT_len(self, len):
        pass

    def forward(self, inputs, sigma):
        sigma = self.embedding(sigma)

        #print(xF.shape)
        x=inputs.unsqueeze(1)
        pyr=x
        x=self.init_conv(x) 
        
        hs=[]
        for i,modules in enumerate(self.downs):
           
            if i<(self.depth-1):
                resnet, downsample, combiner=modules
                x=resnet(x,sigma)
                hs.append(x)
                x=downsample(x)
                pyr=downsample(pyr)
                x=combiner(pyr,x)
                

            elif i==(self.depth-1): #no downsampling in the last layer
                (resnet,)=modules
                x=resnet(x,sigma)
                hs.append(x)

        for modules in self.middle:
            (resnet,) =modules 
            x=resnet(x,sigma)

        pyr=None
        for i,modules in enumerate(self.ups):
            j=self.depth -i-1
            if j>0: 
                resnet, upsample, combiner=modules
                
                skip=hs.pop()
                #print(x.shape, skip.shape)
                x =self.cropconcat(x, skip) #there will be problems here, use cropping if necessary
                x=resnet(x,sigma)
                        
                pyr=combiner(pyr,x)
                
                x=upsample(x)
                pyr=upsample(pyr)
                
            elif j==0: #no upsampling in the last layer
                (resnet,)=modules
                skip=hs.pop()
                x=self.cropconcat(x,skip)
                #x = torch.cat((x, hs.pop()), dim=1)
                x=resnet(x,sigma)
                pyr=combiner(pyr,x)


        #print("end ", x.shape)
        pred=pyr.squeeze(1)
        assert pred.shape==inputs.shape, "bad shapes"
        return pred


class CropAddBlock(nn.Module):

    def forward(self,down_layer, x,  **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        #print(x1_shape,x2_shape)
        height_diff = (x1_shape[2] - x2_shape[2]) // 2


        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff)]
        x = torch.add(down_layer_cropped, x)
        return x

class CropConcatBlock(nn.Module):

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff)]
        x = torch.cat((down_layer_cropped, x),1)
        return x


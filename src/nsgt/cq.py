# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

--
Original matlab code copyright follows:

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

"""

from .nsgfwin_sl import nsgfwin
from .nsdual import nsdual
from .nsgtf import nsgtf
from .nsigtf import nsigtf
from .util import calcwinrange
from .fscale import OctScale
from math import ceil
import torch


class NSGT:
    def __init__(self, scale, fs, Ls, real=True, matrixform=False, reducedform=0, multichannel=False, measurefft=False, multithreading=False, dtype=torch.float32, device="cpu"):
        assert fs > 0
        assert Ls > 0
        assert 0 <= reducedform <= 2

        self.scale = scale
        self.fs = fs
        self.Ls = Ls
        self.real = real
        self.measurefft = measurefft
        self.multithreading = multithreading
        self.reducedform = reducedform

        self.device = torch.device(device)
        
        self.frqs,self.q = scale()

        # calculate transform parameters
        self.g,rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.Ls, sliced=False, dtype=dtype, device=self.device)

        if real:
            assert 0 <= reducedform <= 2
            sl = slice(reducedform,len(self.g)//2+1-reducedform)
        else:
            sl = slice(0,None)

        # coefficients per slice
        self.ncoefs = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))        

        if matrixform:
            if self.reducedform:
                rm = self.M[self.reducedform:len(self.M)//2+1-self.reducedform]
                self.M[:] = rm.max()
            else:
                self.M[:] = self.M.max()
    
        if multichannel:
            self.channelize = lambda s: s
            self.unchannelize = lambda s: s
        else:
            self.channelize = lambda s: (s,)
            self.unchannelize = lambda s: s[0]

        # calculate shifts
        self.wins,self.nn = calcwinrange(self.g, rfbas, self.Ls, device=self.device)
        # calculate dual windows
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, device=self.device)
        
        self.fwd = lambda s: nsgtf(s, self.g, self.wins, self.nn, self.M, real=self.real, reducedform=self.reducedform, measurefft=self.measurefft, multithreading=self.multithreading, device=self.device, matrixform=matrixform)
        self.bwd = lambda c: nsigtf(c, self.gd, self.wins, self.nn, self.Ls, real=self.real, reducedform=self.reducedform,matrixform=matrixform, measurefft=self.measurefft, multithreading=self.multithreading, device=self.device)
        
    @property
    def coef_factor(self):
        return float(self.ncoefs)/self.Ls
    
    @property
    def slice_coefs(self):
        return self.ncoefs
    
    def forward(self, s):
        'transform'
        s = self.channelize(s)
        #c = list(map(self.fwd, s))
        c = self.fwd(s)
        #return self.unchannelize(c)
        return c

    def backward(self, c):
        'inverse transform'
        c = self.channelize(c)
        #s = list(map(self.bwd,c))
        s = self.bwd(c)
        return self.unchannelize(s)
    
class CQ_NSGT(NSGT):
    def __init__(self, fmin, fmax, bins, fs, Ls, real=True, matrixform=False, reducedform=0, multichannel=False, measurefft=False, multithreading=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0
        
        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins

        scale = OctScale(fmin, fmax, bins)
        NSGT.__init__(self, scale, fs, Ls, real, matrixform=matrixform, reducedform=reducedform, multichannel=multichannel, measurefft=measurefft, multithreading=multithreading)

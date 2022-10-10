# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

--

% Perfect reconstruction sliCQ

% right now, even slice length (sl_len) is required. Parameters are the
% same as NSGTF plus slice length, minimal required window length, 
% Q-factor variation, and test run parameters.
"""

import torch
import numpy as np
from itertools import cycle, chain, tee
from math import ceil

from .slicing import slicing
from .unslicing import unslicing
from .nsdual import nsdual
from .nsgfwin_sl import nsgfwin
from .nsgtf import nsgtf_sl
from .nsigtf import nsigtf_sl
from .util import calcwinrange
from .fscale import OctScale
from .reblock import reblock


def overlap_add_slicq(slicq, flatten=False):
    # proper 50% overlap-add
    if not flatten:
        nb_samples, nb_slices, nb_channels, nb_f_bins, nb_m_bins = slicq.shape

        window = nb_m_bins
        hop = window//2 # 50% overlap window

        ncoefs = nb_slices*nb_m_bins//2 + hop
        out = torch.zeros((nb_samples, nb_channels, nb_f_bins, ncoefs), dtype=slicq.dtype, device=slicq.device)

        ptr = 0

        for i in range(nb_slices):
            out[:, :, :, ptr:ptr+window] += slicq[:, i, :, :, :]
            ptr += hop

        return out
    # flatten adjacent slices, just for demo purposes
    else:
        slicq = slicq.permute(0, 2, 3, 1, 4)
        out = torch.flatten(slicq, start_dim=-2, end_dim=-1)
        return out


def arrange(cseq, fwd, device="cpu"):
    if type(cseq) == torch.Tensor:
        M = cseq.shape[-1]

        if fwd:
            odd_mid = M//4
            even_mid = 3*M//4
        else:
            odd_mid = 3*M//4
            even_mid = M//4

        # odd indices
        cseq[1::2, :, :, :] = torch.cat((cseq[1::2, :, :, odd_mid:], cseq[1::2, :, :, :odd_mid]), dim=-1)

        # even indices
        cseq[::2, :, :, :] = torch.cat((cseq[::2, :, :, even_mid:], cseq[::2, :, :, :even_mid]), dim=-1)
    elif type(cseq) == list:
        for i, cseq_tsor in enumerate(cseq):
            cseq[i] = arrange(cseq_tsor, fwd, device)
    else:
        raise ValueError(f'unsupported type {type(cseq)}')

    return cseq


def starzip(iterables):
    def inner(itr, i):
        for t in itr:
            yield t[i]
    iterables = iter(iterables)
    it = next(iterables)  # we need that to determine the length of one element
    iterables = chain((it,), iterables)
    return [inner(itr, i) for i,itr in enumerate(tee(iterables, len(it)))]


#@profile
def chnmap_forward(gen, seq, device="cpu"):
    chns = starzip(seq) # returns a list of generators (one for each channel)

    # fuck generators, use a tensor
    chns = [list(x) for x in chns]

    f_slices = torch.empty(len(chns[0]), len(chns), len(chns[0][0]), dtype=torch.float32, device=torch.device(device))

    for i, chn in enumerate(chns):
        for j, sig in enumerate(chn):
            f_slices[j, i, :] = sig

    ret = gen(f_slices)

    return ret


class NSGT_sliced(torch.nn.Module):
    def __init__(self, scale, sl_len, tr_area, fs,
                 min_win=16, Qvar=1,
                 real=False, recwnd=False, matrixform=False, reducedform=0,
                 multichannel=False,
                 measurefft=False,
                 multithreading=False,
                 dtype=torch.float32,
                 device="cpu"):
        assert fs > 0
        assert sl_len > 0
        assert tr_area >= 0
        assert sl_len > tr_area*2
        assert min_win > 0
        assert 0 <= reducedform <= 2

        assert sl_len%4 == 0
        assert tr_area%2 == 0

        super(NSGT_sliced, self).__init__()

        self.device = torch.device(device)

        self.sl_len = sl_len
        self.tr_area = tr_area
        self.fs = fs
        self.real = real
        self.measurefft = measurefft
        self.multithreading = multithreading
        self.userecwnd = recwnd
        self.reducedform = reducedform
        self.multichannel = multichannel

        self.scale = scale
        self.frqs,self.q = self.scale()

        self.g,self.rfbas,self.M = nsgfwin(self.frqs, self.q, self.fs, self.sl_len, sliced=True, min_win=min_win, Qvar=Qvar, dtype=dtype, device=self.device)
        
#        print "rfbas",self.rfbas/float(self.sl_len)*self.fs
        if real:
            assert 0 <= reducedform <= 2
            sl = slice(reducedform,len(self.g)//2+1-reducedform)
        else:
            sl = slice(0,None)

        self.fbins_actual = sl.stop

        # coefficients per slice
        self.ncoefs = max(int(ceil(float(len(gii))/mii))*mii for mii,gii in zip(self.M[sl],self.g[sl]))

        self.matrixform = matrixform
        
        if self.matrixform:
            if self.reducedform:
                rm = self.M[self.reducedform:len(self.M)//2+1-self.reducedform]
                self.M[:] = rm.max()
            else:
                self.M[:] = self.M.max()

        if multichannel:
            self.channelize = lambda seq: seq
            self.unchannelize = lambda seq: seq
        else:
            self.channelize = lambda seq: ((it,) for it in seq)
            self.unchannelize = lambda seq: (it[0] for it in seq)

        self.wins,self.nn = calcwinrange(self.g, self.rfbas, self.sl_len, device=self.device)
        
        self.gd = nsdual(self.g, self.wins, self.nn, self.M, device=self.device)
        self.setup_lambdas()
        
    def setup_lambdas(self):
        self.fwd = lambda fc: nsgtf_sl(fc, self.g, self.wins, self.nn, self.M, real=self.real, reducedform=self.reducedform, matrixform=self.matrixform, measurefft=self.measurefft, multithreading=self.multithreading, device=self.device)
        self.bwd = lambda cc: nsigtf_sl(cc, self.gd, self.wins, self.nn, self.sl_len ,real=self.real, reducedform=self.reducedform, matrixform=self.matrixform, measurefft=self.measurefft, multithreading=self.multithreading, device=self.device)

    def _apply(self, fn):
        super(NSGT_sliced, self)._apply(fn)
        self.wins = [fn(w) for w in self.wins]
        self.g = [fn(g) for g in self.g]
        self.device = self.g[0].device
        self.setup_lambdas()

    @property
    def coef_factor(self):
        return float(self.ncoefs)/self.sl_len
    
    @property
    def slice_coefs(self):
        return self.ncoefs
    
    #@profile
    def forward(self, sig):
        'transform - s: iterable sequence of sequences' 

        sig = self.channelize(sig)

        # Compute the slices (zero-padded Tukey window version)
        f_sliced = slicing(sig, self.sl_len, self.tr_area, device=self.device)

        cseq = chnmap_forward(self.fwd, f_sliced, device=self.device)

        cseq = arrange(cseq, True, device=self.device)
    
        cseq = self.unchannelize(cseq)

        return cseq

    #@profile
    def backward(self, cseq, length):
        'inverse transform - c: iterable sequence of coefficients'
        cseq = self.channelize(cseq)

        cseq = arrange(cseq, False, device=self.device)

        frec_sliced = self.bwd(cseq)

        # Glue the parts back together
        ftype = float if self.real else complex
        sig = unslicing(frec_sliced, self.sl_len, self.tr_area, dtype=ftype, usewindow=self.userecwnd, device=self.device)

        sig = list(self.unchannelize(sig))[2:]

        # convert to tensor
        ret = next(reblock(sig, length, fulllast=False, multichannel=self.multichannel, device=self.device))

        return ret


class CQ_NSGT_sliced(NSGT_sliced):
    def __init__(self, fmin, fmax, bins, sl_len, tr_area, fs, min_win=16, Qvar=1, real=False, recwnd=False, matrixform=False, reducedform=0, multichannel=False, measurefft=False, multithreading=False):
        assert fmin > 0
        assert fmax > fmin
        assert bins > 0

        self.fmin = fmin
        self.fmax = fmax
        self.bins = bins  # bins per octave

        scale = OctScale(fmin, fmax, bins)
        NSGT_sliced.__init__(self, scale, sl_len, tr_area, fs, min_win, Qvar, real, recwnd, matrixform=matrixform, reducedform=reducedform, multichannel=multichannel, measurefft=measurefft, multithreading=multithreading)

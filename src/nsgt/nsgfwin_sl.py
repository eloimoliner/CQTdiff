# -*- coding: utf-8

"""
Thomas Grill, 2011-2016
http://grrrr.org/nsgt

--
Original matlab code comments follow:

NSGFWIN.M
---------------------------------------------------------------
 [g,rfbas,M]=nsgfwin(fmin,bins,sr,Ls) creates a set of windows whose
 centers correspond to center frequencies to be
 used for the nonstationary Gabor transform with varying Q-factor. 
---------------------------------------------------------------

INPUT : fmin ...... Minimum frequency (in Hz)
        bins ...... Vector consisting of the number of bins per octave
        sr ........ Sampling rate (in Hz)
        Ls ........ Length of signal (in samples)

OUTPUT : g ......... Cell array of window functions.
         rfbas ..... Vector of positions of the center frequencies.
         M ......... Vector of lengths of the window functions.

AUTHOR(s) : Monika Dörfler, Gino Angelo Velasco, Nicki Holighaus, 2010

COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
http://nuhag.eu/
Permission is granted to modify and re-distribute this
code in any manner as long as this notice is preserved.
All standard disclaimers apply.

EXTERNALS : firwin
"""

import numpy as np
from .util import hannwin, blackharr, blackharrcw
from math import ceil
from warnings import warn
from itertools import chain
import torch


def nsgfwin(f, q, sr, Ls, sliced=True, min_win=4, Qvar=1, dowarn=True, dtype=np.float64, device="cpu"):
    nf = sr/2.

    lim = np.argmax(f > 0)
    if lim != 0:
        # f partly <= 0 
        f = f[lim:]
        q = q[lim:]
            
    lim = np.argmax(f >= nf)
    if lim != 0:
        # f partly >= nf 
        f = f[:lim]
        q = q[:lim]
    
    assert len(f) == len(q)
    assert np.all((f[1:]-f[:-1]) > 0)  # frequencies must be increasing
    assert np.all(q > 0)  # all q must be > 0
    
    qneeded = f*(Ls/(8.*sr))
    if np.any(q >= qneeded) and dowarn:
        warn("Q-factor too high for frequencies %s"%",".join("%.2f"%fi for fi in f[q >= qneeded]))
    
    fbas = f
    lbas = len(fbas)
    
    frqs = np.concatenate(((0.,),fbas,(nf,)))
    
    fbas = np.concatenate((frqs,sr-frqs[-2:0:-1]))

    # at this point: fbas.... frequencies in Hz
    
    fbas *= float(Ls)/sr
    
#    print "fbas",fbas
    
    # Omega[k] in the paper
    if sliced:
        M = np.zeros(fbas.shape, dtype=float)
        M[0] = 2*fbas[1]
        M[1] = fbas[1]/q[0] #(2**(1./bins[0])-2**(-1./bins[0]))
        for k in chain(range(2,lbas),(lbas+1,)):
            M[k] = fbas[k+1]-fbas[k-1]
        M[lbas] = fbas[lbas]/q[lbas-1] #(2**(1./bins[-1])-2**(-1./bins[-1]))
#        M[lbas+1] = fbas[lbas]/q[lbas-1] #(2**(1./bins[-1])-2**(-1./bins[-1]))
        M[lbas+2:2*(lbas+1)] = M[lbas:0:-1]
#        M[-1] = M[1]
        M *= Qvar/4.
        M = np.round(M).astype(int)
        M *= 4        
    else:
        M = np.zeros(fbas.shape, dtype=int)
        M[0] = np.round(2*fbas[1])
        for k in range(1,2*lbas+1):
            M[k] = np.round(fbas[k+1]-fbas[k-1])
        M[-1] = np.round(Ls-fbas[-2])
        
    np.clip(M, min_win, np.inf, out=M)

#    print "M",list(M)
    
    if sliced: 
        g = [blackharr(m, device=device).to(dtype) for m in M]
    else:
        g = [hannwin(m, device=device).to(dtype) for m in M]
    
    if sliced:
        for kk in (1,lbas+2):
            if M[kk-1] > M[kk]:
                g[kk-1] = torch.ones(M[kk-1], dtype=g[kk-1].dtype, device=torch.device(device))
                g[kk-1][M[kk-1]//2-M[kk]//2:M[kk-1]//2+int(ceil(M[kk]/2.))] = hannwin(M[kk], device=device)
        
        rfbas = np.round(fbas/2.).astype(int)*2
    else:
        fbas[lbas] = (fbas[lbas-1]+fbas[lbas+1])/2
        fbas[lbas+2] = Ls-fbas[lbas]
        rfbas = np.round(fbas).astype(int)
        
#    print "rfbas",rfbas
#    print "g",g

    return g,rfbas,M

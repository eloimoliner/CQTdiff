# -*- coding: utf-8

"""
Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

--
Original matlab code comments follow:

NSDUAL.M - Nicki Holighaus 02.02.11

Computes (for the painless case) the dual frame corresponding to a given 
non-stationary Gabor frame specified by the windows 'g' and time shifts
'shift'.

Note, the time shifts corresponding to the dual window sequence is the
same as the original shift sequence and as such already given.

This routine's output can be used to achieve reconstruction of a signal 
from its non-stationary Gabor coefficients using the inverse 
non-stationary Gabor transform 'nsigt'.

More information on Non-stationary Gabor transforms
can be found at:

http://www.univie.ac.at/nonstatgab/

minor edit by Gino Velasco 23.02.11
"""

import numpy as np
import torch

from .util import chkM


def nsdual(g, wins, nn, M=None, device="cpu"):
    M = chkM(M,g)

    # Construct the diagonal of the frame operator matrix explicitly
    x = torch.zeros((nn,), dtype=float, device=torch.device(device))
    for gi,mii,sl in zip(g, M, wins):
        xa = torch.square(torch.fft.fftshift(gi))
        xa *= mii

        #print('xa: {0} {1} {2}'.format(xa.shape, xa.dtype, xa.device))
        #print('x: {0} {1} {2}'.format(x.shape, x.dtype, x.device))
        x[sl] += xa

        # could be more elegant...
#        (w1a,w1b),(w2a,w2b) = sl
#        x[w1a] += xa[:w1a.stop-w1a.start]
#        xa = xa[w1a.stop-w1a.start:]
#        x[w1b] += xa[:w1b.stop-w1b.start]
#        xa = xa[w1b.stop-w1b.start:]
#        x[w2a] += xa[:w2a.stop-w2a.start]
#        xa = xa[w2a.stop-w2a.start:]
#        x[w2b] += xa[:w2b.stop-w2b.start]
##        xa = xa[w1b.stop-w1b.start:]

    # Using the frame operator and the original window sequence, compute 
    # the dual window sequence
#    gd = [gi/N.fft.ifftshift(N.hstack((x[wi[0][0]],x[wi[0][1]],x[wi[1][0]],x[wi[1][1]]))) for gi,wi in izip(g,wins)]
    gd = [gi/torch.fft.ifftshift(x[wi]) for gi,wi in zip(g,wins)]
    return gd

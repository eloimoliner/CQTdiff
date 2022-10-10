# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

import numpy as np

class Scale:
    dbnd = 1.e-8
    
    def __init__(self, bnds):
        self.bnds = bnds
        
    def __len__(self):
        return self.bnds
    
    def Q(self, bnd=None):
        # numerical differentiation (if self.Q not defined by sub-class)
        if bnd is None:
            bnd = np.arange(self.bnds)
        return self.F(bnd)*self.dbnd/(self.F(bnd+self.dbnd)-self.F(bnd-self.dbnd))
    
    def __call__(self):
        f = np.array([self.F(b) for b in range(self.bnds)],dtype=float)
        q = np.array([self.Q(b) for b in range(self.bnds)],dtype=float)
        return f,q

    def suggested_sllen_trlen(self, sr):
        f,q = self()

        Ls = int(np.ceil(max((q*8.*sr)/f)))

        # make sure its divisible by 4
        Ls = Ls + -Ls % 4

        sllen = Ls

        trlen = sllen//4
        trlen = trlen + -trlen % 2 # make trlen divisible by 2

        return sllen, trlen


class OctScale(Scale):
    def __init__(self, fmin, fmax, bpo, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bpo: bands per octave (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        lfmin = np.log2(fmin)
        lfmax = np.log2(fmax)
        bnds = int(np.ceil((lfmax-lfmin)*bpo))+1
        Scale.__init__(self, bnds+beyond*2)
        odiv = (lfmax-lfmin)/(bnds-1)
        lfmin_ = lfmin-odiv*beyond
        lfmax_ = lfmax+odiv*beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = np.sqrt(self.pow2n)/(self.pow2n-1.)/2.
        
    def F(self, bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else np.arange(self.bnds))
    
    def Q(self, bnd=None):
        return self.q


class LogScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        Scale.__init__(self, bnds+beyond*2)
        lfmin = np.log2(fmin)
        lfmax = np.log2(fmax)
        odiv = (lfmax-lfmin)/(bnds-1)
        lfmin_ = lfmin-odiv*beyond
        lfmax_ = lfmax+odiv*beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = np.sqrt(self.pow2n)/(self.pow2n-1.)/2.
        
    def F(self, bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else np.arange(self.bnds))
    
    def Q(self, bnd=None):
        return self.q


class VQLogScale(Scale):
    def __init__(self, fmin, fmax, bnds, gamma=0, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param gamma: decrease q at low frequencies with an offset
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        Scale.__init__(self, bnds+beyond*2)
        lfmin = np.log2(fmin)
        lfmax = np.log2(fmax)
        odiv = (lfmax-lfmin)/(bnds-1)
        lfmin_ = lfmin-odiv*beyond
        lfmax_ = lfmax+odiv*beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        #self.q = np.sqrt(self.pow2n)/(self.pow2n-1.)/2.
        self.gamma = gamma
        
    def F(self, bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else np.arange(self.bnds)) + self.gamma


class LinScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        self.df = float(fmax-fmin)/(bnds-1)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)-self.df*beyond
        if self.fmin <= 0:
            raise ValueError("Frequencies must be > 0.")
        self.fmax = float(fmax)+self.df*beyond

    def F(self, bnd=None):
        return (bnd if bnd is not None else np.arange(self.bnds))*self.df+self.fmin

    def Q(self, bnd=None):
        return self.F(bnd)/(self.df*2)


def hz2mel(f):
    "\cite{shannon:2003}"
    return np.log10(f/700.+1.)*2595.


def mel2hz(m):
    "\cite{shannon:2003}"
    return (np.power(10.,m/2595.)-1.)*700.


class MelScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        mmin = hz2mel(fmin)
        mmax = hz2mel(fmax)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.mbnd = (mmax-mmin)/(bnds-1)  # mels per band
        self.mmin = mmin-self.mbnd*beyond
        self.mmax = mmax+self.mbnd*beyond
        
    def F(self, bnd=None):
        if bnd is None:
            bnd = np.arange(self.bnds)
        return mel2hz(bnd*self.mbnd+self.mmin)

    def Q1(self, bnd=None): # obviously not exact
        if bnd is None:
            bnd = np.arange(self.bnds)
        mel = bnd*self.mbnd+self.mmin
        odivs = (np.exp(mel/-1127.)-1.)*(-781.177/self.mbnd)
        pow2n = np.power(2, 1./odivs)
        return np.sqrt(pow2n)/(pow2n-1.)/2.
    

def hz2bark(f):
    #       HZ2BARK         Converts frequencies Hertz (Hz) to Bark
    #
    b = 6 * np.arcsinh(f/600)
    return b


def bark2hz(b):
    #       BARK2HZ         Converts frequencies Bark to Hertz (HZ)
    #
    f = 600*np.sinh(b/6)
    return f


class BarkScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        bmin = hz2bark(fmin)
        bmax = hz2bark(fmax)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.bbnd = (bmax-bmin)/(bnds-1)  # mels per band
        self.bmin = bmin-self.bbnd*beyond
        self.bmax = bmax+self.bbnd*beyond

    def F(self, bnd=None):
        if bnd is None:
            bnd = np.arange(self.bnds)
        return bark2hz(bnd*self.bbnd+self.bmin)


'''
a toy frequency scale based on powers of two
to demonstrate the flexibility of the sliCQT
'''
class Pow2Scale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        self.start = 0
        while 2**self.start < fmin:
            self.start += 1

        Scale.__init__(self, bnds-self.start+beyond*2)

        if 2**(bnds-1) >= fmax:
            raise ValueError(f'too many frequency bands!')

    def F(self, bnd=None):
        if bnd is None:
            bnd = np.arange(self.bnds)
        return 2**(bnd+self.start)

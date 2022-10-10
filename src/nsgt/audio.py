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
import subprocess as sp
import os.path
import re
import sys
from functools import reduce

try:
    from scikits.audiolab import Sndfile, Format
except:
    Sndfile = None
    
def sndreader(sf, blksz=2**16, dtype=np.float32):
    if dtype is float:
        dtype = np.float64 # scikits.audiolab needs numpy types
    if blksz < 0:
        blksz = sf.nframes
    if sf.channels > 1: 
        channels = lambda s: s.T
    else:
        channels = lambda s: s.reshape((1,-1))
    for offs in range(0, sf.nframes, blksz):
        data = sf.read_frames(min(sf.nframes-offs, blksz), dtype=dtype)
        yield channels(data)
    
def sndwriter(sf, blkseq, maxframes=None):
    written = 0
    for b in blkseq:
        b = b.T
        if maxframes is not None: 
            b = b[:maxframes-written]
        sf.write_frames(b)
        written += len(b)

def findfile(fn, path=os.environ['PATH'].split(os.pathsep), matchFunc=os.path.isfile):
    for dirname in path:
        candidate = os.path.join(dirname, fn)
        if matchFunc(candidate):
            return candidate
    return None


class SndReader:
    def __init__(self, fn, sr=None, chns=None, blksz=2**16, dtype=np.float32):
        fnd = False
                
        if not fnd and (Sndfile is not None):
            try:
                sf = Sndfile(fn)
            except IOError:
                pass
            else:
                if (sr is None or sr == sf.samplerate) and (chns is None or chns == sf.channels):
                    # no resampling required
                    self.channels = sf.channels
                    self.samplerate = sf.samplerate
                    self.frames = sf.nframes
                
                    self.rdr = sndreader(sf, blksz, dtype=dtype)
                    fnd = True                
        
        if not fnd:
            ffmpeg = findfile('ffmpeg') or findfile('avconv') or findfile('ffmpeg.exe') or findfile('avconv.exe')
            if ffmpeg is not None:
                pipe = sp.Popen([ffmpeg,'-i', fn,'-'],stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
                fmtout = pipe.stderr.read()
                if (sys.version_info > (3, 0)):
                    fmtout = fmtout.decode()
                m = re.match(r"^(ffmpeg|avconv) version.*Duration: (\d\d:\d\d:\d\d.\d\d),.*Audio: (.+), (\d+) Hz, (.+), (.+), (\d+) kb/s", " ".join(fmtout.split('\n')))
                if m is not None:
                    self.samplerate = int(m.group(4)) if not sr else int(sr)
                    self.channels = {'mono':1, '1 channels (FL+FR)':1, 'stereo':2}[m.group(5)] if not chns else chns
                    dur = reduce(lambda x,y: x*60+y, list(map(float, m.group(2).split(':'))))
                    self.frames = int(dur*self.samplerate)  # that's actually an estimation, because of potential resampling with round-off errors
                    pipe = sp.Popen([ffmpeg,
                        '-i', fn,
                        '-f', 'f32le',
                        '-acodec', 'pcm_f32le',
                        '-ar', str(self.samplerate),
                        '-ac', str(self.channels),
                        '-'],
    #                    bufsize=self.samplerate*self.channels*4*50,
                        stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
                    def rdr():
                        while True:
                            data = pipe.stdout.read(blksz*4)
                            if len(data) == 0:
                                break
                            yield np.fromstring(data, dtype=dtype).reshape((-1, self.channels)).T
                    self.rdr = rdr()
                    fnd = True                
                
        if not fnd:
            raise IOError("Format not usable")
        
    def __call__(self):
        return self.rdr


class SndWriter:
    def __init__(self, fn, samplerate, filefmt='wav', datafmt='pcm16', channels=1):
        fmt = Format(filefmt, datafmt)
        self.sf = Sndfile(fn, mode='w', format=fmt, channels=channels, samplerate=samplerate)
        
    def __call__(self, sigblks, maxframes=None):
        sndwriter(self.sf, sigblks, maxframes=None)


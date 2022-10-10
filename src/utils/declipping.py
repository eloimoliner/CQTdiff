import numpy as np
from scipy.optimize import fsolve

def get_clip_value_from_SDR(seg, SDRdesired):

    def find_clip_value(thresh, x, SDRtarget):
        xclipped=np.clip(x, -thresh, thresh)
        sdr=20*np.log10(np.linalg.norm(x)/(np.linalg.norm(x-xclipped)+1e-7));
        return np.abs(sdr-SDRtarget)

    clip_value=fsolve(find_clip_value, 0.1, args=(seg.cpu().numpy(), SDRdesired))

    return clip_value[0]

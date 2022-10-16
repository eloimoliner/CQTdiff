"""
Train a diffusion model on images.
"""
import os
import hydra
import logging
import torch
import torchaudio
torch.cuda.empty_cache() 
import soundfile as sf

from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
import numpy as np
from datetime import date

#from learner import Learner
#from model import UNet
import soundfile as sf


from tqdm import tqdm

import scipy.signal


def run(args):
    """Loads all the modules and starts the sampling
        
    Args: args: Hydra dictionary

    """

    #some preparation of the hydra args
    args = OmegaConf.structured(OmegaConf.to_yaml(args))


    dirname = os.path.dirname(__file__)

    #define the path where weights will be loaded and audio samples and other logs will be saved
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    mode=args.inference.mode
    plot_animation=True
    
    if mode=="unconditional":
        from src.experimenters.exp_unconditional import Exp_Unconditional
        exp=Exp_Unconditional(args, plot_animation)
    elif mode=="bandwidth_extension":
        from src.experimenters.exp_bandwidth_extension import Exp_BWE
        exp=Exp_BWE(args, plot_animation)
    elif mode=="declipping":
        from src.experimenters.exp_declipping import Exp_Declipping
        exp=Exp_Declipping(args, plot_animation)
    elif mode=="inpainting":
        from src.experimenters.exp_inpainting import Exp_Inpainting
        exp=Exp_Inpainting(args, plot_animation)
    elif mode=="compressive_sensing":
        from src.experimenters.exp_comp_sens import Exp_CompSens
        exp=Exp_CompSens(args, plot_animation)
    elif mode=="phase_retrieval":
        from src.experimenters.exp_phase_retrieval import Exp_PhaseRetrieval
        exp=Exp_PhaseRetrieval(args, plot_animation)

    print(args.dset.name)
        
    import src.utils.setup as utils_setup
    if mode!="unconditional":
        print("load test set")
        test_set = utils_setup.get_test_set_for_sampling(args)
    else:
        test_set=None
    print(test_set)

    if mode!="unconditional":
        for i, (original, filename) in enumerate(tqdm(test_set)):
    
            n=os.path.splitext(filename[0])[0]
    
            #process the file if its output has not been already generated
            if not(os.path.exists(os.path.join(exp.path_reconstructed,n+".wav"))):
    
                 print("Sampling on mode ",mode," with the file ",filename[0])
         
                 original=original.float()
         
                 #resampling if the sampling rate is not 22050, this information should be given on the resmaple_factor
                 print(original.shape)
                 if args.resample_factor!=1:
                     S=args.resample_factor
                     if S>2.1 and S<2.2:
                         #resampling 48k to 22.05k
                         resample=torchaudio.transforms.Resample(160*2,147) #I use 2**12 as an arbitrary number, as we don't care about the sampling frequency of the latents
                     else:
                         N=int(args.audio_len*S)
                         resample=torchaudio.transforms.Resample(N,args.audio_len)
                     original=resample(original)
         
                 #we need to split the track somehow, here we use only one chunk
                 
                 #seg=original[...,10*args.sample_rate:(10*args.sample_rate+args.audio_len)]
                 seg=original
                
                 print(seg.shape, original.shape)
    
                 exp.conduct_experiment(seg, n)
    else:

        #unconditional mode
        for i in range(args.inference.unconditional.num_samples):
            exp.conduct_experiment(str(i))
             

def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="conf", config_name="conf")
def main(args):
    #try:
        _main(args)
    #except Exception:
        #logger.exception("Some error happened")
    #    os._exit(1)

if __name__ == "__main__":
    main()

import os
import torch
from src.sampler import SamplerPhaseRetrieval
from src.experimenters.exp_base import Exp_Base
import src.utils.bandwidth_extension as utils_bwe
import src.utils.logging as utils_logging

class Exp_PhaseRetrieval(Exp_Base):
    def __init__(self, args, plot_animation):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """ 
        super().__init__(args)
        self.__plot_animation=plot_animation
        self.sampler=SamplerInpainting(self.model, self.diff_parameters, self.args, args.inference.alpha, order=2, data_consistency=not(args.inference.no_replace), rid=self.__plot_animation)


        self.__filter_final=utils_bwe.get_FIR_lowpass(100,10000,1,self.args.sample_rate)
        self.__filter_final=self.__filter_final.to(self.device)

        #clipping specific
        n="spectrograms"+"_"+str(self.args.inference.phase_retrieval.win_size)
        self.path_degraded=os.path.join(self.path_sampling, n) #path for the clipped outputs
        #ensure the path exists
        if not os.path.exists(self.path_degraded):
            os.makedirs(self.path_degraded)
    
        self.path_original=os.path.join(self.path_sampling, "original") #this will need a better organization
        if not os.path.exists(self.path_original):
            os.makedirs(self.path_original)
    
        #path where the output will be saved
        n="reconstructed"+"_"+str(self.args.inference.phase_retrieval.win_size)
        self.path_reconstructed=os.path.join(self.path_sampling, n) #path for the clipped outputs
        #ensure the path exists
        if not os.path.exists(self.path_reconstructed):
            os.makedirs(self.path_reconstructed)

        self.stft_args={
                        "win_size":self.args.inference.phase_retrieval.win_size,
                        "hop_size":self.args.inference.phase_retrieval.hop_size,
                       }

    def apply_stft(self,x, stft_args):
        win_size=stft_args["win_size"]
        hop_size=stft_args["hop_size"]
        window=torch.hamming_window(window_length=win_size).to(x.device)
        x2=torch.cat((x, torch.zeros(x.shape[0],win_size ).to(x.device)),-1)
        X=torch.stft(x2, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
        Y=torch.sqrt(X[...,0]**2 + X[...,1]**2)
        return Y

    def conduct_experiment(self,seg, name):
        #assuming now that the input has the expected shape
        assert seg.shape[-1]==self.args.audio_len

        seg=seg.to(self.device)
        audio_path=utils_logging.write_audio_file(seg, self.args.sample_rate, name, self.path_original+"/")

     
            
        #compute spectrogram
        y=self.apply_stft(seg, self.stft_args)

        utils_logging.plot_mag_spectrogram(y,refr=1, path=self.path_degraded, name=name)      

        #input("stop")
        if self.__plot_animation:
            x_hat, data_denoised, t=self.sampler.predict_pr(y,self.stft_args)
            fig=utils_logging.diffusion_spec_animation(self.path_reconstructed,  data_denoised, t, self.args.stft, name="animation"+name)
        else:
            x_hat=self.sampler.predict_pr(y,self.stft_args)
           
        #apply low pass filter to remove annoying artifacts at the nyquist frequency. I should try to fix this issue in future work
        x_hat=utils_bwe.apply_low_pass(x_hat, self.__filter_final) 

        #save reconstructed audio file
        audio_path=utils_logging.write_audio_file(x_hat, self.args.sample_rate, name, self.path_reconstructed+"/")
        

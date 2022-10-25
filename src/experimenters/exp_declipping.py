import os
import torch
from src.sampler import SamplerDeclipping
from src.experimenters.exp_base import Exp_Base
import src.utils.bandwidth_extension as utils_bwe
import src.utils.declipping as utils_declipping
import src.utils.logging as utils_logging

class Exp_Declipping(Exp_Base):
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
        self.sampler=SamplerDeclipping(self.model, self.diff_parameters, self.args, args.inference.xi, order=2, data_consistency=args.inference.data_consistency, rid=self.__plot_animation)


        self.__filter_final=utils_bwe.get_FIR_lowpass(100,10000,1,self.args.sample_rate)
        self.__filter_final=self.__filter_final.to(self.device)

        #clipping specific
        n="clipped"+"_"+str(self.args.inference.declipping.SDR)
        self.path_degraded=os.path.join(self.path_sampling, n) #path for the clipped outputs
        #ensure the path exists
        if not os.path.exists(self.path_degraded):
            os.makedirs(self.path_degraded)
    
        self.path_original=os.path.join(self.path_sampling, "original") #this will need a better organization
        if not os.path.exists(self.path_original):
            os.makedirs(self.path_original)
    
        #path where the output will be saved
        n="declipped"+"_"+str(self.args.inference.declipping.SDR)
        self.path_reconstructed=os.path.join(self.path_sampling, n) #path for the clipped outputs
        #ensure the path exists
        if not os.path.exists(self.path_reconstructed):
            os.makedirs(self.path_reconstructed)


    def conduct_experiment(self,seg, name):
        #assuming now that the input has the expected shape
        print(seg.shape[-1])
        assert seg.shape[-1]==self.args.audio_len

        seg=seg.to(self.device)
        audio_path=utils_logging.write_audio_file(seg, self.args.sample_rate, name, self.path_original+"/")
        print(audio_path)

     
        clip_value=utils_declipping.get_clip_value_from_SDR(seg,self.args.inference.declipping.SDR)
            
        #apply clip
        y=torch.clip(seg,min=-clip_value, max=clip_value)

        #save clipped audio file
        audio_path_clipped=utils_logging.write_audio_file(y, self.args.sample_rate, name, self.path_degraded+"/")
        print(audio_path)

        #input("stop")
        if self.__plot_animation:
            x_hat, data_denoised, t=self.sampler.predict_declipping(y,clip_value)
            fig=utils_logging.diffusion_CQT_animation(self.path_reconstructed,  data_denoised, t, self.args.stft, name="animation"+name, compression_factor=8)
        else:
            x_hat=self.sampler.predict_declipping(y,clip_value)
           
        #apply low pass filter to remove annoying artifacts at the nyquist frequency. I should try to fix this issue in future work
        x_hat=utils_bwe.apply_low_pass(x_hat, self.__filter_final, "firwin")

        #save reconstructed audio file
        audio_path_result=utils_logging.write_audio_file(x_hat, self.args.sample_rate, name, self.path_reconstructed+"/")
        print(audio_path)
        if self.__plot_animation:
            return audio_path_clipped, audio_path_result, fig
        else:
            return audio_path_clipped, audio_path_result
        

import os
import torch
from datetime import date
from src.sampler import SamplerInpainting
from src.experimenters.exp_base import Exp_Base
import src.utils.bandwidth_extension as utils_bwe
import src.utils.bandwidth_extension as utils_inpainting
import src.utils.logging as utils_logging

class Exp_Inpainting(Exp_Base):
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
        self.sampler=SamplerInpainting(self.model, self.diff_parameters, self.args, args.inference.xi, order=2, data_consistency=not(args.inference.no_replace), rid=self.__plot_animation)

        today=date.today() 
        self.path_sampling=os.path.join(args.model_dir,self.args.inference.mode+today.strftime("%d/%m/%Y"))
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

        self.__filter_final=utils_bwe.get_FIR_lowpass(100,10000,1,self.args.sample_rate)
        self.__filter_final=self.__filter_final.to(self.device)

        #clipping specific
        n="masked"+"_"+str(self.args.inference.inpainting.gap_length)
        self.path_degraded=os.path.join(self.path_sampling, n) #path for the clipped outputs
        #ensure the path exists
        if not os.path.exists(self.path_degraded):
            os.makedirs(self.path_degraded)
    
        self.path_original=os.path.join(self.path_sampling, "original") #this will need a better organization
        if not os.path.exists(self.path_original):
            os.makedirs(self.path_original)
    
        #path where the output will be saved
        n="inpainted"+"_"+str(self.args.inference.inpainting.gap_length)
        self.path_reconstructed=os.path.join(self.path_sampling, n) #path for the clipped outputs
        #ensure the path exists
        if not os.path.exists(self.path_reconstructed):
            os.makedirs(self.path_reconstructed)

        gap=int(self.args.inference.inpainting.gap_length*self.args.sample_rate/1000)      
        self.mask=torch.ones((1,self.args.audio_len)).to(self.device) #assume between 5 and 6s of total length

        if self.args.inference.inpainting.start_gap_idx ==None:
            #the gap is placed at the center
            start_gap_index=int(self.args.audio_len//2 - gap//2) 
        else:
            start_gap_index=int(self.args.inference.inpainting.start_gap_idx*self.args.sample_rate/1000)

        self.mask[...,start_gap_index:(start_gap_index+gap)]=0



    def conduct_experiment(self,seg, name):
        #assuming now that the input has the expected shape
        assert seg.shape[-1]==self.args.audio_len

        seg=seg.to(self.device)
        audio_path=utils_logging.write_audio_file(seg, self.args.sample_rate, name, self.path_original+"/")

     
            
        #apply mask
        y=self.mask*seg

        #save clipped audio file
        audio_path=utils_logging.write_audio_file(y, self.args.sample_rate, n, self.path_degraded+"/")

        #input("stop")
        if self.__plot_animation:
            x_hat, data_denoised, t=self.sampler.predict_inpainting(y,self.mask)
            fig=utils_logging.diffusion_spec_animation(self.path_reconstructed,  data_denoised, t, self.args.stft, name="animation"+n)
        else:
            x_hat=self.sampler.predict_inpainting(y,self.mask)
           
        #apply low pass filter to remove annoying artifacts at the nyquist frequency. I should try to fix this issue in future work
        x_hat=utils_bwe.apply_low_pass(x_hat, self.__filter_final) 

        #save reconstructed audio file
        audio_path=utils_logging.write_audio_file(x_hat, args.sample_rate, n, self.path_reconstructed+"/")
        

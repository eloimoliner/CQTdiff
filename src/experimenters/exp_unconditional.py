import os
import torch
from src.sampler import Sampler
from src.experimenters.exp_base import Exp_Base
import src.utils.bandwidth_extension as utils_bwe
import src.utils.logging as utils_logging

class Exp_Unconditional(Exp_Base):
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

        self.sampler=Sampler(self.model, self.diff_parameters, self.args, 0, order=2, data_consistency=args.inference.data_consistency, rid=self.__plot_animation)

        self.__filter_final=utils_bwe.get_FIR_lowpass(100,10000,1,self.args.sample_rate)
        self.__filter_final=self.__filter_final.to(self.device)

        self.path_degraded=None
        #ensure the path exists
        self.path_original=None
    
        #path where the output will be saved
        n="sampled"
        self.path_reconstructed=os.path.join(self.path_sampling, n) #path for the clipped outputs
        #ensure the path exists
        if not os.path.exists(self.path_reconstructed):
            os.makedirs(self.path_reconstructed)


    def conduct_experiment(self, name):
        #assuming now that the input has the expected shape

        #input("stop")
        #shape=(self.args.inference.unconditional.num_samples, self.args.inference.audio_len)
        shape=(1, self.args.audio_len)
        if self.__plot_animation:
            x_hat, data_denoised, t=self.sampler.predict_unconditional(shape, self.device)
            fig=utils_logging.diffusion_CQT_animation(self.path_reconstructed,  data_denoised, t, self.args, name="animation"+name, resample_factor=4)
        else:
            x_hat=self.sampler.predict_unconditional(shape,self.device)
           
        #apply low pass filter to remove annoying artifacts at the nyquist frequency. I should try to fix this issue in future work
        x_hat=utils_bwe.apply_low_pass(x_hat, self.__filter_final) 

        #save reconstructed audio file
        audio_path=utils_logging.write_audio_file(x_hat, self.args.sample_rate, name, self.path_reconstructed+"/")
        

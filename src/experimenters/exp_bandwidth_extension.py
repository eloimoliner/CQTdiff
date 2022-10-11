import os
import torch
from src.sampler import SamplerBWE
from src.experimenters.exp_base import Exp_Base
import src.utils.bandwidth_extension as utils_bwe
import src.utils.logging as utils_logging

class Exp_BWE(Exp_Base):
    def __init__(self, args, plot_animation):
        super().__init__(args)
        self.__plot_animation=plot_animation
        self.sampler=SamplerBWE(self.model, self.diff_parameters, self.args, args.inference.xi, order=2, data_consistency=args.inference.data_consistency, rid=self.__plot_animation)


        self.__filter_final=utils_bwe.get_FIR_lowpass(100,10000,1,self.args.sample_rate)
        self.__filter_final=self.__filter_final.to(self.device)

        n="lowpassed"+"_"+str(self.args.inference.declipping.SDR)
        self.path_degraded=os.path.join(self.path_sampling, n) #path for the lowpassed 
        #ensure the path exists
        if not os.path.exists(self.path_degraded):
            os.makedirs(self.path_degraded)
    
        self.path_original=os.path.join(self.path_sampling, "original") #this will need a better organization
        if not os.path.exists(self.path_original):
            os.makedirs(self.path_original)
    
        #path where the output will be saved
        n="bwe"+"_"+str(self.args.inference.bandwidth_extension.filter.fc)+"_"+str(self.args.inference.bandwidth_extension.filter.type)
        self.path_reconstructed=os.path.join(self.path_sampling, n) #path for the clipped outputs
        #ensure the path exists
        if not os.path.exists(self.path_reconstructed):
            os.makedirs(self.path_reconstructed)

        #design the filter
        order=self.args.inference.bandwidth_extension.filter.order
        fc=self.args.inference.bandwidth_extension.filter.fc

        if self.args.inference.bandwidth_extension.filter.type=="firwin":
            beta=self.args.inference.bandwidth_extension.filter.beta
            self.filter=utils_bwe.get_FIR_lowpass(order,fc, beta,self.args.sample_rate)
        elif args.inference.bandwidth_extension.filter.type=="cheby1filtfilt":
            raise NotImplementedError
        elif args.inference.bandwidth_extension.filter.type=="butter_fir":
            raise NotImplementedError
        elif args.inference.bandwidth_extension.filter.type=="cheby1_fir":
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.filter=self.filter.to(self.device)



    def conduct_experiment(self,seg, name):
        #assuming now that the input has the expected shape
        assert seg.shape[-1]==self.args.audio_len

        seg=seg.to(self.device)
        audio_path=utils_logging.write_audio_file(seg, self.args.sample_rate, name, self.path_original+"/")

        #apply filter
        
        y=utils_bwe.apply_low_pass(seg, self.filter) 

        #save clipped audio file
        audio_path=utils_logging.write_audio_file(y, self.args.sample_rate, name, self.path_degraded+"/")

        #input("stop")
        if self.__plot_animation:
            x_hat, data_denoised, t=self.sampler.predict_bwe(y,self.filter)
            fig=utils_logging.diffusion_spec_animation(self.path_reconstructed,  data_denoised, t, self.args.stft, name="animation"+name)
        else:
            x_hat=self.sampler.predict_bwe(y, self.filter)
           
        #apply low pass filter to remove annoying artifacts at the nyquist frequency. I should try to fix this issue in future work
        x_hat=utils_bwe.apply_low_pass(x_hat, self.__filter_final) 

        #save reconstructed audio file
        audio_path=utils_logging.write_audio_file(x_hat, self.args.sample_rate, name, self.path_reconstructed+"/")
        

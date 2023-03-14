import torch
import numpy as np
import src.utils.logging as utils_logging


class VE_Sde_Elucidating():
    """
        Definition of most of the diffusion parameterization, following ( Karras et al., "Elucidating...", 2022)
    """

    def __init__(self, args, sigma_data):
        """
        Args:
            args (dictionary): hydra arguments
            sigma_data (float): precalculated variance of the dataset
        """
        self.sigma_min = args.sigma_min
        self.sigma_max =args.sigma_max
        self.P_mean=args.P_mean
        self.P_std=args.P_std
        self.ro=args.ro
        self.ro_train=args.ro_train
        self.sigma_data=sigma_data #depends on the training data!!
        #parameters stochastic sampling
        self.Schurn=args.Schurn
        self.Stmin=args.Stmin
        self.Stmax=args.Stmax
        self.Snoise=args.Snoise

       

    def get_gamma(self, t): 
        """
        Get the parameter gamma that defines the stochasticity of the sampler
        Args
            t (Tensor): shape: (N_steps, ) Tensor of timesteps, from which we will compute gamma
        """
        N=t.shape[0]
        gamma=torch.zeros(t.shape, device=t.device)
        
        #If desired, only apply stochasticity between a certain range of noises Stmin is 0 by default and Stmax is a huge number by default. (Unless these parameters are specified, this does nothing)
        indexes=torch.logical_and(t>self.Stmin , t<self.Stmax)
         
        #We use Schurn=5 as the default in our experiments
        gamma[indexes]=gamma[indexes]+torch.min(torch.Tensor([self.Schurn/N, 2**(1/2) -1]))
        
        return gamma

    def create_schedule(self,nb_steps):
        """
        Define the schedule of timesteps
        Args:
           nb_steps (int): Number of discretized steps
        """
        i=torch.arange(0,nb_steps+1)
        t=(self.sigma_max**(1/self.ro) +i/(nb_steps-1) *(self.sigma_min**(1/self.ro) - self.sigma_max**(1/self.ro)))**self.ro
        t[-1]=0
        return t


    def sample_ptrain(self,N):
        """
        For training, getting t as a normal distribution, folowing Karras et al. 
        I'm not using this
        Args:
            N (int): batch size
        """
        lnsigma=np.random.randn(N)*self.P_std +self.P_mean
        return np.clip(np.exp(lnsigma),self.sigma_min, self.sigma_max) #not sure if clipping here is necessary, but makes sense to me
    
    def sample_ptrain_alt(self,N):
        """
        For training, getting  t according to the same criteria as sampling
        Args:
            N (int): batch size
        """
        a=np.random.uniform(size=N)
        t=(self.sigma_max**(1/self.ro_train) +a *(self.sigma_min**(1/self.ro_train) - self.sigma_max**(1/self.ro_train)))**self.ro_train
        return t

    def sample_prior(self,shape,sigma):
        """
        Just sample some gaussian noise, nothing more
        Args:
            shape (tuple): shape of the noise to sample, something like (B,T)
            sigma (float): noise level of the noise
        """
        n=torch.randn(shape).to(sigma.device)*sigma
        return n

    def cskip(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        
        """
        return self.sigma_data**2 *(sigma**2+self.sigma_data**2)**-1

    def cout(self,sigma ):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return sigma*self.sigma_data* (self.sigma_data**2+sigma**2)**(-0.5)

    def cin(self, sigma):
        """
        Just one of the preconditioning parameters
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (self.sigma_data**2+sigma**2)**(-0.5)

    def cnoise(self,sigma ):
        """
        preconditioning of the noise embedding
        Args:
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        return (1/4)*torch.log(sigma)

    def lambda_w(self,sigma):
        return (sigma*self.sigma_data)**(-2) * (self.sigma_data**2+sigma**2)
        
    def denoiser(self, x , model, sigma):
        """
        This method does the whole denoising step, which implies applying the model and the preconditioning
        Args:
            x (Tensor): shape: (B,T) Intermediate noisy latent to denoise
            model (nn.Module): Model of the denoiser
            sigma (float): noise level (equal to timestep is sigma=t, which is our default)
        """
        sigma=sigma.unsqueeze(-1)
        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)

        return cskip * x +cout*model(cin*x, cnoise)  #this will crash because of broadcasting problems, debug later!



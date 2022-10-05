import torch
import numpy as np


class VE_Sde_Elucidating():

    def __init__(self, args, sigma_data):
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
        N=t.shape[0]
        gamma=torch.zeros(t.shape)
        
        indexes=torch.logical_and(t>self.Stmin , t<self.Stmax)
         
        gamma[indexes]=gamma[indexes]+torch.min(torch.Tensor([self.Schurn/N, 2**(1/2) -1]))
        
        return gamma

    def create_schedule(self,nb_steps):
        i=torch.arange(0,nb_steps+1)
        t=(self.sigma_max**(1/self.ro) +i/(nb_steps-1) *(self.sigma_min**(1/self.ro) - self.sigma_max**(1/self.ro)))**self.ro
        t[-1]=0
        return t


    def sample_ptrain(self,N):
        lnsigma=np.random.randn(N)*self.P_std +self.P_mean
        return np.clip(np.exp(lnsigma),self.sigma_min, self.sigma_max) #not sure if clipping here is necessary, but makes sense to me
    
    def sample_ptrain_alt(self,N):
        #chhosing t according to the same criteria as sampling
        a=np.random.uniform(size=N)
        t=(self.sigma_max**(1/self.ro_train) +a *(self.sigma_min**(1/self.ro_train) - self.sigma_max**(1/self.ro_train)))**self.ro_train
        return t

    def sample_prior(self,shape,sigma):
        n=torch.randn(shape).to(sigma.device)*sigma
        return n

    def cskip(self, sigma):
        return self.sigma_data**2 *(sigma**2+self.sigma_data**2)**-1

    def cout(self,sigma ):
        return sigma*self.sigma_data* (self.sigma_data**2+sigma**2)**(-0.5)

    def cin(self, sigma):
        return (self.sigma_data**2+sigma**2)**(-0.5)

    def cnoise(self,sigma ):
        return (1/4)*torch.log(sigma)

    def lambda_w(self,sigma):
        return (sigma*self.sigma_data)**(-2) * (self.sigma_data**2+sigma**2)
        
    def denoiser(self, x , model, sigma):
        sigma=sigma.unsqueeze(-1)
        cskip=self.cskip(sigma)
        cout=self.cout(sigma)
        cin=self.cin(sigma)
        cnoise=self.cnoise(sigma)

        return cskip * x +cout*model(cin*x, cnoise)  #this will crash because of broadcasting problems, debug later!



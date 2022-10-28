"""

Main script for training
"""
import os
import hydra

import torch
torch.cuda.empty_cache()

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np

def run(args):
    """Loads all the modules and starts the training
        
    Args:
      args:
        Hydra dictionary

    """
        
    #some preparation of the hydra args
    args = OmegaConf.structured(OmegaConf.to_yaml(args))

    #choose gpu as the device if possible
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirname = os.path.dirname(__file__)

    #define the path where weights will be loaded and saved
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    torch.backends.cudnn.benchmark = True
        
    def worker_init_fn(worker_id):                                                          
        st=np.random.get_state()[2]
        np.random.seed( st+ worker_id)

    print("Training on: ",args.dset.name)

    #prepare the dataset loader
    import src.dataset_loader as dataset
    dataset_train=dataset.TrainDataset( args.dset, args.sample_rate*args.resample_factor,args.audio_len*args.resample_factor)
    train_loader=DataLoader(dataset_train,num_workers=args.num_workers, batch_size=args.batch_size,  worker_init_fn=worker_init_fn)
    train_set = iter(train_loader)
        
    #prepare the model architecture
    
    if args.architecture=="unet_CQT":
        from src.models.unet_cqt import Unet_CQT
        model=Unet_CQT(args, device).to(device)
    elif args.architecture=="unet_STFT":
        from src.models.unet_stft import Unet_STFT
        self.model=Unet_STFT(self.args, self.device).to(self.device)
    elif args.architecture=="unet_1d":
        from src.models.unet_1d import Unet_2d
        self.model=Unet_1d(self.args, self.device).to(self.device)
    else:
        raise NotImplementedError

    #prepare the optimizer

    from src.learner import Learner
    
    learner = Learner(
        args.model_dir, model, train_set,  args, log=args.log
    )

    #start the training
    learner.train()


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="conf", config_name="conf")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()

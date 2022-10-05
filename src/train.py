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

    print("Starting the training")

    args = OmegaConf.structured(OmegaConf.to_yaml(args))

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirname = os.path.dirname(__file__)
    path_experiment = os.path.join(dirname, str(args.model_dir))

    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)

    args.model_dir=path_experiment

    torch.backends.cudnn.benchmark = True
        
    def worker_init_fn(worker_id):                                                          
        st=np.random.get_state()[2]
        np.random.seed( st+ worker_id)

    print("Training on: ",args.dset.name)
    #prepare the dataset
    import dataset_loader as dataset
    dataset_train=dataset.TrainDataset( args.dset, args.sample_rate*args.resample_factor,args.audio_len*args.resample_factor)
    train_loader=DataLoader(dataset_train,num_workers=args.num_workers, batch_size=args.batch_size,  worker_init_fn=worker_init_fn)
    train_set = iter(train_loader)
        
    #prepare the model architecture
    from model import Unet2d
    model=Unet2d(args, device).to(device)

    #prepare the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    from learner import Learner
    learner = Learner(
        args.model_dir, model, train_set, opt, args, log=log
    )

    #start the training
    learner.train()


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="../conf", config_name="conf")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()

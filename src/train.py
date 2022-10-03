"""
Train a diffusion model on images.
"""
import os
import hydra
import logging

import torch
torch.cuda.empty_cache()
import numpy as npp

#from improved_diffusion import dist_util, logger
#from improved_diffusion.image_datasets import load_data
#from improved_diffusion.script_util import (
#    model_and_diffusion_defaults,
#    create_model_and_diffusion,
#    args_to_dict,
#    add_dict_to_argparser,k#)
#import dataset_loader
#import dataset_loader_random
#import dataset_noise_loader
#import toy_dataset

from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from torch.utils.data import DataLoader
import numpy as np

#from learner import Learner
#from model import UNet
from getters import get_sde
import utils
import utils_setup

def run(args):

    print("starting")
    args = OmegaConf.structured(OmegaConf.to_yaml(args))
    #OmegaConf.set_struct(conf, True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    dirname = os.path.dirname(__file__)
    path_experiment = os.path.join(dirname, str(args.model_dir))

    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)

    args.model_dir=path_experiment
    #dist_util.setup_dist()

    #logger.configure()

    torch.backends.cudnn.benchmark = True
        



    #logger.log("creating data loader...")
    def worker_init_fn(worker_id):                                                          
        st=np.random.get_state()[2]
        np.random.seed( st+ worker_id)

    print(args.dset.name)
    #train_set, test_set = utils_setup.get_train_and_test_sets(args)

    #copied
    import dataset_loaders.dataset_maestro_loader as dataset
    dataset_train=dataset.TrainDataset( args.dset, args.sample_rate*args.resample_factor,args.audio_len*args.resample_factor)
    train_loader=DataLoader(dataset_train,num_workers=args.num_workers, batch_size=args.batch_size,  worker_init_fn=worker_init_fn)
    train_set = iter(train_loader)
    if load_testset:

        dataset_test=dataset.ValDataset( args.dset, args.sample_rate*args.resample_factor,args.audio_len*args.resample_factor)
        #print("average std: ", dataset_test.get_std_avg()) #1.03
        #input("stop")
        test_set= DataLoader(dataset_test,num_workers=args.num_workers, shuffle=False, batch_size=1,  worker_init_fn=worker_init_fn)
    else:
        test_set=None
    #copied
    print("dataset loaded")

    #model, learner= utils_setup.get_model_and_learner(args, device, train_set, test_set)
    #copied
    from models.unet_CQT_nobias import Unet2d
    model=Unet2d(args, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    from learners.learner_july import Learner
    learner = Learner(
        args.model_dir, model, train_set,test_set, opt, args, log=log
    )
    #copied


    learner.is_master = True
    #learner.restore_from_checkpoint(params['checkpoint_id'])
    learner.train()


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

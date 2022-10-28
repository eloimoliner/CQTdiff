import os
import torch 
import numpy as np
import glob

from torch.utils.data import DataLoader


def load_ema_weights(model, model_dir):
    checkpoint = torch.load(model_dir, map_location=model.device)

    #print(len(checkpoint['model']), len(checkpoint['ema_weights']))
    try:
      dic_ema = {}
      for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
          dic_ema[key] = tensor
      model.load_state_dict(dic_ema)
    except:
      dic_ema = {}
      i=0
      for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['model'].values()):
          if tensor.requires_grad:
              dic_ema[key]=checkpoint['ema_weights'][i]
              i=i+1
          else:
              dic_ema[key]=tensor     
      model.load_state_dict(dic_ema)
    return model

def load_weights(model, model_dir):
    checkpoint = torch.load(model_dir, map_location=device)

    #print(len(checkpoint['model']), len(checkpoint['ema_weights']))
    try:
      dic_ema = {}
      for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['weights']):
          dic_ema[key] = tensor
      model.load_state_dict(dic_ema)
    except:
      dic_ema = {}
      i=0
      for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['model'].values()):
          if tensor.requires_grad:
              dic_ema[key]=checkpoint['weights'][i]
              i=i+1
          else:
              dic_ema[key]=tensor     
      model.load_state_dict(dic_ema)
    return model

def do_istft(data, win_size=2048, hop_size=512, device="cuda"):
    window=torch.hamming_window(window_length=win_size) #this is slow! consider optimizing
    window=window.to(device)
    data=data.permute(0,3,2,1)
    pred_time=torch.istft(data, win_size, hop_length=hop_size,  window=window, center=False, return_complex=False)
    return pred_time

def remove_high_freqs(noisy, clean=None, f_dim=1025):
    noisy=noisy[:,:,:,0:f_dim] 
    if clean!=None:
        clean=clean[:,:,:,0:f_dim] 
        return noisy, clean
    else:
        return noisy


def do_stft(noisy, clean=None, win_size=2048, hop_size=512, device="cpu", DC=True):
    
    #window_fn = tf.signal.hamming_window

    #win_size=args.stft.win_size
    #hop_size=args.stft.hop_size
    window=torch.hamming_window(window_length=win_size)
    window=window.to(noisy.device)
    noisy=torch.cat((noisy, torch.zeros(noisy.shape[0],win_size).to(noisy.device)),1)
    stft_signal_noisy=torch.stft(noisy, win_size, hop_length=hop_size,window=window,center=False,return_complex=False)
    stft_signal_noisy=stft_signal_noisy.permute(0,3,2,1)
    #stft_signal_noisy=tf.signal.stft(noisy,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
    #stft_noisy_stacked=tf.stack( values=[tf.math.real(stft_signal_noisy), tf.math.imag(stft_signal_noisy)], axis=-1)
    
    if clean!=None:

       # stft_signal_clean=tf.signal.stft(clean,frame_length=win_size, window_fn=window_fn, frame_step=hop_size)
        clean=torch.cat((clean, torch.zeros(clean.shape[0],win_size).to(device)),1)
        stft_signal_clean=torch.stft(clean, win_size, hop_length=hop_size,window=window, center=False,return_complex=False)
        stft_signal_clean=stft_signal_clean.permute(0,3,2,1)
        #stft_clean_stacked=tf.stack( values=[tf.math.real(stft_signal_clean), tf.math.imag(stft_signal_clean)], axis=-1)


        if DC:
            return stft_signal_noisy, stft_signal_clean
        else:
            return stft_signal_noisy[...,1:], stft_signal_clean[...,1:]
    else:

        if DC:
            return stft_signal_noisy
        else:
            return stft_signal_noisy[...,1:]


def worker_init_fn(worker_id):
    st=np.random.get_state()[2]
    np.random.seed( st+ worker_id)
 

def get_test_set_for_sampling(args):
    if args.inference.load.load_mode=="from_directory":
        """
        loading the dataset from a directory, (NOT TESTED YET)
        """
        assert args.inference.load.data_directory!=None
        assert args.inference.load.seg_size!= None
        assert args.inference.load.seg_idx!= None
        files=glob.glob(os.path.join(args.inference.load.data_directory,"*.wav"))
        assert len(files)>0
        import src.dataset_loader as dataset
        dataset_test=dataset.TestDataset_chunks_fromdir(files, args.sample_rate*args.resample_factor,args.audio_len*args.resample_factor, args.inference.load.seg_idx, return_name=True)
        print(dataset_test)
        test_set= DataLoader(dataset_test,num_workers=args.num_workers, shuffle=False, batch_size=1,  worker_init_fn=worker_init_fn)
        return test_set

    elif args.inference.load.load_mode=="maestro_test":
        assert args.inference.load.seg_size!= None
        assert args.inference.load.seg_idx!= None
        test_set=None
        if args.dset.name=="maestro":
            import src.dataset_loader as dataset
            dataset_test=dataset.TestDataset_chunks(args.dset, args.sample_rate*args.resample_factor,args.audio_len*args.resample_factor, args.inference.load.seg_idx, return_name=True)
            test_set= DataLoader(dataset_test,num_workers=args.num_workers, shuffle=False, batch_size=1,  worker_init_fn=worker_init_fn)
        return test_set

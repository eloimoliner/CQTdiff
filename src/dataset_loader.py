import ast

#import tensorflow as tf
import random
import scipy.signal
import os
import numpy as np
import soundfile as sf
import math
import pandas as pd
import glob
from tqdm import tqdm
import torch

#generator function. It reads the csv file with pandas and loads the largest audio segments from each recording. If extend=False, it will only read the segments with length>length_seg, trim them and yield them with no further processing. Otherwise, if the segment length is inferior, it will extend the length using concatenative synthesis.



class ValDataset(torch.utils.data.Dataset):

    def __init__(self, dset_args, fs=44100, seg_len=131072):

        path=dset_args.path
        years=dset_args.years_val

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]=="validation"]

        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x), na_action='ignore')

        val_samples=filelist.to_list()

        print("Loading clean files")
        data_clean_loaded=[]
        for ff in tqdm(range(0,len(val_samples))):  
            data_clean, samplerate = sf.read(val_samples[ff])
            if samplerate!=fs: 
                print(samplerate, fs)
                print("!!!!WRONG SAMPLE RATe!!!")
            #Stereo to mono
            if len(data_clean.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
            #normalize
            #data_clean=data_clean/np.max(np.abs(data_clean))
            data_clean_loaded.append(data_clean)
            del data_clean
    
        #framify data clean files
        print("Framifying clean files")
        seg_len=int(seg_len)
        self.segments_clean=[]
        num_segments_max=500 #I do that to avoid having too large testing
        j=0
        for file in tqdm(data_clean_loaded):
            #print(file) 
            #framify  arguments: seg_len, hop_size
            hop_size=int(seg_len)# no overlap
    
            num_frames=np.floor(len(file)/hop_size - seg_len/hop_size +1) 
            
            pointer=0
            for i in range(0,int(num_frames)):
                #print(i, num_frames)
                segment=file[pointer:pointer+int(seg_len)]
                pointer=pointer+hop_size
                segment=segment.astype('float32')

                self.segments_clean.append(segment)
                j+=1
                if j>500: break
    
        del data_clean_loaded
        
        #scales=np.random.uniform(-6,4,len(self.segments_clean))

    def __len__(self):
        #print(len(self.segments_clean))
        return len(self.segments_clean)

    def __getitem__(self, idx):
        return self.segments_clean[idx]

class TestDataset_fullpiece(torch.utils.data.Dataset):

    def __init__(self, dset_args, fs=44100, seg_len=131072, return_name=False):

        self.return_name=return_name
        path=dset_args.path
        years=dset_args.years_test
        print(years)

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]=="test"]

        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x), na_action='ignore')

        val_samples=filelist.to_list()

        print("Loading clean files")
        self.data_clean_loaded=[]
        if return_name: self.filenames=[]
        #maxsamples=2
        for ff in tqdm(range(0,len(val_samples))):  

            #if ff>=maxsamples: break
            if return_name:
                name=os.path.normpath(val_samples[ff]).split(os.path.sep)
                self.filenames.append(name[-1])

            data_clean, samplerate = sf.read(val_samples[ff])
            if samplerate!=fs: 
                print(samplerate, fs)
                print("!!!!WRONG SAMPLE RATe!!!")
            #Stereo to mono
            if len(data_clean.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
            #normalize
            #data_clean=data_clean/np.max(np.abs(data_clean))
            self.data_clean_loaded.append(data_clean)
            del data_clean
    
        #framify data clean files
        
        #scales=np.random.uniform(-6,4,len(self.segments_clean))

    def __len__(self):
        #print(len(self.segments_clean))
        return len(self.data_clean_loaded)

    def __getitem__(self, idx):
        if self.return_name:
            return self.data_clean_loaded[idx], self.filenames[idx]
        else:
            return self.data_clean_loaded[idx]

class TestDataset(torch.utils.data.Dataset):

    def __init__(self, dset_args, fs=44100, seg_len=131072):

        path=dset_args.path
        years=dset_args.years_test

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]=="test"]

        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x), na_action='ignore')

        val_samples=filelist.to_list()

        print("Loading clean files")
        data_clean_loaded=[]
        for ff in tqdm(range(0,len(val_samples))):  
            data_clean, samplerate = sf.read(val_samples[ff])
            if samplerate!=fs: 
                print(samplerate, fs)
                print("!!!!WRONG SAMPLE RATe!!!")
            #Stereo to mono
            if len(data_clean.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
            #normalize
            #data_clean=data_clean/np.max(np.abs(data_clean))
            data_clean_loaded.append(data_clean)
            del data_clean
    
        #framify data clean files
        print("Framifying clean files")
        seg_len=int(seg_len)
        self.segments_clean=[]
        for file in tqdm(data_clean_loaded):
            #print(file) 
            #framify  arguments: seg_len, hop_size
            hop_size=int(seg_len)# no overlap
    
            num_frames=np.floor(len(file)/hop_size - seg_len/hop_size +1) 
            
            pointer=0
            for i in range(0,int(num_frames)):
                #print(i, num_frames)
                segment=file[pointer:pointer+int(seg_len)]
                pointer=pointer+hop_size
                segment=segment.astype('float32')

                self.segments_clean.append(segment)
    
        del data_clean_loaded
        
        #scales=np.random.uniform(-6,4,len(self.segments_clean))

    def __len__(self):
        #print(len(self.segments_clean))
        return len(self.segments_clean)

    def __getitem__(self, idx):
        return self.segments_clean[idx]
        

#Train dataset object

class TrainDataset(torch.utils.data.IterableDataset):
    def __init__(self, dset_args,  fs=44100, seg_len=131072, seed=42 ):
        super(TrainDataset).__init__()
        random.seed(seed)
        np.random.seed(seed)
        path=dset_args.path
        years=dset_args.years

        metadata_file=os.path.join(path,"maestro-v3.0.0.csv")
        metadata=pd.read_csv(metadata_file)

        metadata=metadata[metadata["year"].isin(years)]
        metadata=metadata[metadata["split"]=="train"]
        filelist=metadata["audio_filename"]

        filelist=filelist.map(lambda x:  os.path.join(path,x)     , na_action='ignore')


        self.train_samples=filelist.to_list()
       
        self.seg_len=int(seg_len)
        self.fs=fs

    def __iter__(self):
        while True:
            num=random.randint(0,len(self.train_samples)-1)
            #for file in self.train_samples:  
            file=self.train_samples[num]
            data, samplerate = sf.read(file)
            assert(samplerate==self.fs, "wrong sampling rate")
            data_clean=data
            #Stereo to mono
            if len(data.shape)>1 :
                data_clean=np.mean(data_clean,axis=1)
    
            #normalize
            #no normalization!!
            #data_clean=data_clean/np.max(np.abs(data_clean))
         
            #framify data clean files
            num_frames=np.floor(len(data_clean)/self.seg_len) 
            if num_frames>4:
                for i in range(8):
                    #get 8 random batches to be a bit faster
                    idx=np.random.randint(0,len(data_clean)-self.seg_len)
                    segment=data_clean[idx:idx+self.seg_len]
                    segment=segment.astype('float32')
                    #b=np.mean(np.abs(segment))
                    #segment= (10/(b*np.sqrt(2)))*segment #default rms  of 0.1. Is this scaling correct??
                     
                    #let's make this shit a bit robust to input scale
                    #scale=np.random.uniform(1.75,2.25)
                    #this way I estimage sigma_data (after pre_emph) to be around 1
                   
                    #segment=10.0**(scale) *segment
                    yield  segment
            else:
                pass
         


             



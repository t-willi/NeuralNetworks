import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import glob
import wfdb # to read ecgs of Physionet
from wfdb import processing # to normalize
class ECGDataSimple(Dataset):
    def __init__(self, data_dir, norm=[-1,1], cropping=None, transform=None):
        
        #self.all_ecg_files = []
        self.norm = norm
        self.norm_lb = norm[0]
        self.norm_ub = norm[1]
        self.cropping = cropping
        
        #for data_dir in data_dirs:
        self.ecg_files = glob.glob(data_dir + "/*/*.dat")
        # self.all_ecg_files = self.all_ecg_files + ecg_files

        self.transform = transform

    def __len__(self):
        return len(self.ecg_files)

    def __getitem__(self, idx):
        
        # Get file name of asc file
        file_name = self.ecg_files[idx][:-4] #remove .dat at the end 
        
        ecg_signals = wfdb.rdsamp(file_name)[0]

        if self.norm:
            processing.normalize_bound(ecg_signals, lb=self.norm_lb, ub=self.norm_ub)

        ecg_signals = torch.tensor(ecg_signals) # convert to tensor
        
      
        ecg_signals = ecg_signals.float()
        
        #print(ecg_signals.shape)
        #ecg_signals = ecg_signals  / self.norm_num # Normalizing aplitude of voltage levels 

        #cropping
        if self.cropping:
            ecg_signals = ecg_signals[self.cropping[0]:self.cropping[1], :]
        
        # Transposing the ecg signals
        ecg_signals = ecg_signals.t() 
        #ecg_signals = ecg_signals.unsqueeze(0)
     
  

        if self.transform:
            ecg_signals = self.transform(ecg_signals)
            #sample = self.transform(sample["ecg_signals"])

        # Return sample at the end
        sample = {'ecg_signals': ecg_signals}

        return sample
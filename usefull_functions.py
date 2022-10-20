import glob
from pathlib import Path
from random import shuffle
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

def hello():
    return("Hello")
    
def find_max_min(data_dir=None):
    """
    This takes a directory of pd.read_csv readable data and intterates trough all,
    finding the global max and min
    """
    from tqdm.auto import tqdm
    max_list=[]
    min_list=[]
    #data_dir_train = "/content/data/train_data/sub_data/train"
    files=glob.glob(data_dir + '/*.asc')
    #files_train=glob.glob(data_dir_train + '/*.asc')
    #files=[*files_test,*files_train]
    for file in tqdm(files):
      temp_df=pd.read_csv(file,sep=" ",names=["1","2","v1","v2","v3","v4","v5","v6"])
      maximum=temp_df.max().max()
      minimum=temp_df.min().min()
      max_list.append(maximum)
      min_list.append(minimum)
      maximum=max(max_list)
      minimum=min(min_list)
    return(maximum,minimum)
  
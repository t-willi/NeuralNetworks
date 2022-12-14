import glob
import pandas as pd
import torch
class Custom_dataset_PTB():
    def __init__(self, data_dir,max_value=33,column=3,split=True,target="train",size=1):
      #get all files from directory loaded in all_files list
      self.column=column
      self.max_value=max_value
      self.size=size
      #should shuffle the data here?
      self.files = glob.glob(data_dir + '/*.csv')
      self.len=int((len(self.files))*self.size)
      #print(f"len:{self.len}")
      self.cut1=int(self.len*0.8)
      #print(f"cut1:{self.cut1}")
      self.cut2=int(self.len*0.9)
      #print(f"cut2:{self.cut2}")
      self.train_files=self.files[0:self.cut1]
      self.test_files=self.files[self.cut1:self.cut2]
      self.val_files=self.files[self.cut2:self.len]
      self.target=target
      self.split=split

    def __len__(self):
      if self.split is True:
        if self.target is "train":
          return len(self.train_files)
        if self.target is "test":
          return len(self.test_files)
        if self.target is "val":
          return len(self.val_files)
      if self.split is not True:
        return len(self.files)

    def __getitem__(self,idx):
      header=["I", "II", "III", "aVF", "aVR", "aVL", "v1", "v2", "v3", "v4", "v5", "v6"]
      #turn list of dataframes into Tensor
      if self.split is True:
        if self.target is "train":
          temp_df=pd.read_csv(self.train_files[idx],index_col=0,header=0,names=header)
        if self.target is "test":
          temp_df=pd.read_csv(self.test_files[idx],index_col=0,header=0,names=header)
        if self.target is "val":
          temp_df=pd.read_csv(self.val_files[idx],index_col=0,header=0,names=header)
      if self.split is not True:
          temp_df=pd.read_csv(self.files[idx],index_col=0,header=0,names=header)
      temp_df/=self.max_value
      #load input tensor
      temp_list_in=temp_df.loc[:,["II","v1","v2","v3","v4","v5","v6"]].values
      #temp_list_in=normalize([temp_list_in], norm="max")
      temp_tensor_in = torch.tensor(temp_list_in,dtype=torch.float32).T
      return temp_tensor_in


def make_loader(dataset,batch_size):
  from torch.utils.data import DataLoader
  loader = DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True
                      )
  return loader
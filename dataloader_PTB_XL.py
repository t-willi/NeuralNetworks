import glob
import pandas as pd
import torch
import wfdb 
from wfdb import processing # to normalize
class Custom_dataset_PTB():
    def __init__(self, data_dir,max_value=5011,column=3,split=True,target="train",size=1):
      #get all files from directory loaded in all_files list
      self.column=column
      self.max_value=max_value
      self.size=size
      #should shuffle the data here?
      self.files = glob.glob(data_dir + "/*/*.dat")
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
      #header = ["1","2","v1","v2","v3","v4","v5","v6"]
      header=["I", "II", "III", "aVF", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
      #turn list of dataframes into Tensor
      if self.split is True:
        if self.target is "train":
          file_name = self.train_files[idx][:-4]
          temp_df=pd.DataFrame(wfdb.rdsamp(file_name)[0], columns = header)
        if self.target is "test":
          file_name = self.test_files[idx][:-4]
          temp_df=pd.DataFrame(wfdb.rdsamp(file_name)[0], columns = header)
        if self.target is "val":
          file_name = self.val_files[idx][:-4]
          temp_df=pd.DataFrame(wfdb.rdsamp(file_name)[0], columns = header)

      
      temp_list_in=temp_df.iloc[:,0]
      #temp_list_in=normalize([temp_list_in], norm="max")
      temp_tensor_in = torch.tensor(temp_list_in,dtype=torch.float32)
      temp_tensor_in=temp_tensor_in.unsqueeze(0)
      #load label Tensor
      temp_list_out=temp_df.iloc[:,[1,6,7,8,9,10,11]].values
      #temp_list_out=normalize([temp_list_out], norm="max")
      temp_tensor_out=torch.tensor(temp_list_out,dtype=torch.float32)
      temp_tensor_out=temp_tensor_out.unsqueeze(0)
      temp_tensor_out=torch.permute(temp_tensor_out,(0,2,1))
      #combine input and label and output
      temp_tensor_pair= temp_tensor_in,temp_tensor_out
      return temp_tensor_pair


def make_loader(dataset,batch_size):
  from torch.utils.data import DataLoader
  loader = DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True
                      )
  return loader

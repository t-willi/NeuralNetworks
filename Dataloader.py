import glob
import pandas as pd
import torch
class Custom_dataset(Dataset):
  
    def __init__(self, data_dir,max_value=5011,column=3,split=False,target="train"):
      #get all files from directory loaded in all_files list
      self.column=column
      self.max_value=max_value
      #should shuffle the data here?
      self.files = glob.glob(data_dir + '/*.asc')
      self.len=len(self.files)
      self.cut1=int(self.len*0.8)
      self.cut2=int(self.len*0.9)
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
      header = ["1","2","v1","v2","v3","v4","v5","v6"]
      #turn list of dataframes into Tensor
      if self.split is True:
        if self.target is "train":
          temp_df=pd.read_csv(self.train_files[idx],sep=" ", names = header)
        if self.target is "test":
          temp_df=pd.read_csv(self.test_files[idx],sep=" ", names = header)
        if self.target is "val":
          temp_df=pd.read_csv(self.val_files[idx],sep=" ", names = header)
      if self.split is not True:
        temp_df=pd.read_csv(self.files[idx],sep=" ", names = header)
      temp_df/=self.max_value
      #load input tensor
      
      temp_list_in=temp_df.iloc[:,0]
      #temp_list_in=normalize([temp_list_in], norm="max")
      temp_tensor_in = torch.tensor(temp_list_in,dtype=torch.float32)
      temp_tensor_in=temp_tensor_in.unsqueeze(0)
      #load label Tensor
      temp_list_out=temp_df.iloc[:,1:9].values
      #temp_list_out=normalize([temp_list_out], norm="max")
      temp_tensor_out=torch.tensor(temp_list_out,dtype=torch.float32)
      temp_tensor_out=temp_tensor_out.unsqueeze(0)
      temp_tensor_out=torch.permute(temp_tensor_out,(0,2,1))
      #combine input and label and output
      temp_tensor_pair= temp_tensor_in,temp_tensor_out
      return temp_tensor_pair

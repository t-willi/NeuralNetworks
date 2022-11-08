import pathlib
from pathlib import Path
current_position=pathlib.PurePath(__file__)
dir=current_position.parent
main_folder=dir.joinpath("main_folder")
Path(main_folder).mkdir(parents=False,exist_ok = True)
#create folder and directory for artifacts
artifact_dir=main_folder.joinpath("artifacts")
Path(artifact_dir).mkdir(parents=False,exist_ok = True)
#create folder and directory for train_data
train_dir=main_folder.joinpath("train_dir")
Path(train_dir).mkdir(parents=False,exist_ok = True)
#create folder and directory for ecg_files
ecg_dir=main_folder.joinpath("ecg")
Path(ecg_dir).mkdir(parents=False,exist_ok = True)
#create folder and directory for saved model sate dicts
model_dir=main_folder.joinpath("model")
Path(model_dir).mkdir(parents=False,exist_ok = True)


# !pip install wandb
# !pip install ecg_plot
import wandb
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import torch.optim as optim
from random import shuffle
from tqdm.auto import tqdm
import requests
import zipfile
from pathlib import Path

if torch.cuda.is_available()==True:
  device="cuda:0"
else:
  device ="cpu"
wandb.login(key="7a8ee9d41cc2d51eb77fd795e14f74a215e63c2d")
api = wandb.Api()
artifact = api.artifact('ecg_simula/setup_weights and biases/ecg_25000.zip:v0')
artifact.download(artifact_dir)
torch.manual_seed(42)


# previous_model = api.artifact('ecg_simula/AE_pTOP_LS_optim/Model:v8', type='Model')
# previous_model.download()

def request(path=None,name=None):
  import requests
  from pathlib import Path
  request = requests.get(path)
  name=name+".py"
  with open(name,"wb") as f:
    f.write(request.content)

unzip_git_dir="https://raw.githubusercontent.com/t-willi/NeuralNetworks/main/unzip.py"
Dataloader_git_dir="https://raw.githubusercontent.com/t-willi/NeuralNetworks/main/dataset_and_loader.py"
#model_git_dir="https://raw.githubusercontent.com/t-willi/NeuralNetworks/main/Networks.py"
get_pred_no_reshape_git_dir="https://raw.githubusercontent.com/t-willi/NeuralNetworks/main/get_predictions_no_reshape.py"
get_pred_git_dir="https://raw.githubusercontent.com/t-willi/NeuralNetworks/main/get_predictions.py"
plt_ECG_git_dir="https://raw.githubusercontent.com/t-willi/NeuralNetworks/main/plot_ECG.py"

#this was changed
request(unzip_git_dir,"Unzip")
from Unzip import unzip

request(Dataloader_git_dir,"dataset_and_loader")
from dataset_and_loader import Custom_dataset as CD
from dataset_and_loader import make_loader as ml

#Download and instantialize model
# request(model_git_dir,"Networks")
# from Networks import ECG_stacked_AE
# model=ECG_AE_v1()

#download prediction generator
request(get_pred_no_reshape_git_dir,"get_predictions")
from get_predictions import get_pred
#download ECG plotter
request(plt_ECG_git_dir,"plot_ECG")
from plot_ECG import plotECG

artifact_dir_str=str(artifact_dir.joinpath("ecg_25000.zip"))
unzip(save_path=train_dir,zip_path=artifact_dir_str,reload=True)

import torch
import pandas as pd

def get_pred(dataset=None,model=None):
  """
  Function takes a Tensor Dataset as input,first a random file from the dataset is selected,
  then the Tensor pair is recombined and shaped into a df-->df_Input. 
  X is used afterwards as input into the model. The predictions are safed as --> df_output.
  Both dataframes are now unscaled by 5011, the max value of the whole dataset.
  Then a tuple pair of input and output is returned.
  """
  import random
  limit=len(dataset)
  rand_idx=random.randint(0,limit)
  X,y=dataset[rand_idx]
  #need to combine tensors to make dataframe for plotting input and output side by side
  full_tensor=torch.cat((X,y.squeeze()))
  full_tensor=full_tensor.numpy()
  df_input=pd.DataFrame(full_tensor).T
  df_input.columns = ["R1","R2","Rv1","Rv2","Rv3","Rv4","Rv5","Rv6"]
  model.to("cpu")
  model.eval()
  with torch.inference_mode():
    output=model(X)
  output=output.detach().numpy()
  output=output.squeeze().T
  #unscale data
  df_output = pd.DataFrame(output,columns=["F2","Fv1","Fv2","Fv3","Fv4","Fv5","Fv6"])*5011
  df_input=df_input*5011
  return df_input,df_output


import pandas as pd
from pathlib import Path
def plotECG(df1=None,df2=None,title=None,pad_df2=True,path=None):
  """
  takes two dataframes with identical columns, concats them and plots them as ecg using ecg_plot
  it also takes the first column of df1 and ads it to df1 if pad_df2 is True
  """
  index=["real1","realR2","realv1","realv2","realv3","realv4","realv5","realv6","real_lead1",
         "pred2","predv1","predv2","predv3","predv4","predv5","predv6"]

  ecg_path=path
  if Path(ecg_path).is_dir():
      print(f"{ecg_path} directory exists.")
  else:
      print(f"Did not find {ecg_path} directory, creating one...")
      Path(ecg_path).mkdir(parents=True, exist_ok=True)
  import ecg_plot
  if pad_df2 is True:
    if len(df1.columns)>len(df2.columns):
      df2.insert(0, 'real_lead1', df1["R1"])
  frames=[df1/1000,df2/1000]
  combined_df=pd.concat(frames,axis=1,join="outer",)
  ecg_plot.plot(combined_df.values.T, sample_rate = 500,title = title,
                     lead_index = index )
  ecg_plot.save_as_png('ecg',str(ecg_path)+"/")
  return combined_df



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class Transpose1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=11, upsample=None, output_padding=1):
        super(Transpose1dLayer, self).__init__()
        self.upsample = upsample
        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_pad = kernel_size // 2
        self.reflection_pad = nn.ConstantPad1d(reflection_pad, value=0)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.Conv1dTrans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        if self.upsample:
            #x = torch.cat((x, in_feature), 1)
            return self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        else:
            return self.Conv1dTrans(x)


class Pulse2pulseGenerator(nn.Module):
    def __init__(self,latent_dim=100, post_proc_filt_len=512,upsample=True):
        super(Pulse2pulseGenerator, self).__init__()
        # "Dense" is the same meaning as fully connection.
        stride = 4
        if upsample:
            stride = 1
            upsample = 5
        # if upsample is anything but none Transpose1dLayer will do
        # self.conv1d(self.reflection_pad(self.upsample_layer(x)))
        # which is a 1d convolution on padded and upsampled data x
        self.deconv_1 = Transpose1dLayer(250 , 250, 25, stride, upsample=upsample)
        self.deconv_2 = Transpose1dLayer(250, 150, 25, stride, upsample=upsample)
        self.deconv_3 = Transpose1dLayer(150, 50, 25, stride, upsample=upsample)
        self.deconv_4 = Transpose1dLayer( 50, 25, 25, stride, upsample=2)
        self.deconv_5 = Transpose1dLayer( 25, 10, 25, stride, upsample=upsample)
        self.deconv_6 = Transpose1dLayer(  10, 7, 25, stride, upsample=2)


        #new convolutional layers
        self.conv_1 = nn.Conv1d(1, 10, 25, stride=2, padding=25 // 2)
        self.conv_2 = nn.Conv1d(10, 25, 25, stride=5, padding= 25 // 2)
        self.conv_3 = nn.Conv1d(25, 50 , 25, stride=2, padding= 25 // 2)
        self.conv_4 = nn.Conv1d(50, 150 , 25, stride=5, padding= 25 // 2)
        self.conv_5 = nn.Conv1d(150, 250 , 25, stride=5, padding= 25 // 2)
        self.conv_6 = nn.Conv1d(250, 250 , 25, stride=5, padding= 25 // 2)
        self.flatt = nn.Flatten()
        self.linear1 = nn.Linear(500,100)
        self.linear2 = nn.Linear(100,500)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, LS=False):
        self.LS=LS
        if x.ndim==2:
          x=x.unsqueeze(0)
        x = F.leaky_relu(self.conv_1(x)) #(1,1,5000 --> 1, 10, 2500)
        x = F.leaky_relu(self.conv_2(x)) #( --> 1, 25, 500)
        x = F.leaky_relu(self.conv_3(x)) #(--> 1, 50, 250)
        x = F.leaky_relu(self.conv_4(x)) # --> 1, 150, 50)
        x = F.leaky_relu(self.conv_5(x)) #(--> 1, 250, 10)
        x = F.leaky_relu(self.conv_6(x)) #(--> 1, 250, 2)-->flatten into (1,500)), then to linear ((1,100)), and then back
        x = self.flatt(x) # (1,500)
        LS = self.linear1(x) #(1,100)
        if self.LS is True:
          return LS
        x = self.linear2(LS) #(1,500)
        zero_dim=x.shape[0]
        x=torch.reshape(x,(zero_dim,250,2)) #1(1,250,2)
        x = F.relu(self.deconv_1(x)) #(--> 1, 250, 10)
        x = F.relu(self.deconv_2(x)) #(--> 1, 150, 50)
        x = F.relu(self.deconv_3(x)) #( --> 1, 50, 250)
        x = F.relu(self.deconv_4(x)) #(--> 1, 25, 500)
        x = F.relu(self.deconv_5(x)) #(--> 1, 10, 2500)
        x = torch.tanh(self.deconv_6(x)) #(1, 7, 5000)
        x=x.squeeze()
        return x

model=Pulse2pulseGenerator().to(device)
# PATH="/content/artifacts/Model:v8/model"
# model.load_state_dict(torch.load(PATH))

import glob
import pandas as pd
import torch
class Custom_dataset():
    def __init__(self, data_dir,max_value=5011,column=3,split=True,target="train",size=1):
      #get all files from directory loaded in all_files list
      self.column=column
      self.max_value=max_value
      self.size=size
      #should shuffle the data here?
      string_data_dir=str(data_dir)
      self.files = glob.glob(string_data_dir+ '/*.asc')
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
      #temp_tensor_out=temp_tensor_out.unsqueeze(0)
      temp_tensor_out=torch.permute(temp_tensor_out,(1,0))
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

import wandb

#wandb.login()

# 🐝 initialise a wandb run
config = dict(
    epochs=1001,
    batch_size=32,
    learning_rate=0.0001,)  #learing rate from puls to puls paper


def model_pipeline(hyperparameters,model=model):
    # tell wandb to get started
    wandb.init(project="AE_pTOP_serverrun_synthetic_data", config=hyperparameters)
    # access all HPs through wandb.config, so logging matches execution!
    config = wandb.config
    # make the model, data, and optimization problem
    train_loader, val_loader,test_dataset, criterion, optimizer,val_dataset = make(config)
    # and use them to train the model
    train(model, train_loader,val_loader,test_dataset, criterion, optimizer,val_dataset,config)
    return model

def make(config):
    # Make the data
    print("making data")
    data_dir=train_dir
    train_dataset = Custom_dataset(data_dir=data_dir,split=True,target="train",size=1)
    val_dataset = Custom_dataset(data_dir=data_dir,split=True,target="val",size=1)
    test_dataset = Custom_dataset(data_dir=data_dir,split=True,target="test",size=1)
    train_loader = ml(train_dataset, batch_size=config.batch_size)
    val_loader = ml(val_dataset, batch_size=config.batch_size)
    

    # Make the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return train_loader, val_loader,test_dataset, criterion, optimizer,val_dataset

def train(model, train_loader,val_loader,test_dataset, criterion, optimizer,val_dataset, config):
  # Tell wandb to watch what the model gets up to: gradients, weights, and more!
  wandb.watch(model, criterion, log="all")
  for epoch in (range(config.epochs)):
    train_loss=0
    for batch,(X,y) in (enumerate(train_loader)):
      # Forward pass ➡
      X, y = X.to(device), y.to(device)
      #print(f"shape of input{x.shape},shape of label_y{y.shape}") 
      model.train()
      output=model(X)
      #print(f"shape of model_output_raw{output.shape}") 
      # output=torch.reshape(output,(config.batch_size, 1, 7, 5000))
      loss = criterion(output,y)
      train_loss += loss
      # Backward pass ⬅
      optimizer.zero_grad()
      loss.backward()
      # Step with optimizer
      optimizer.step()
    #average loss per batch
    train_loss /= len(train_loader)


    val_loss = 0
    model.eval()
    with torch.inference_mode():
      for batch,(X,y) in enumerate(val_loader):
        #print("doing test loop")
        X, y = X.to(device), y.to(device)
        val_pred = model(X)
        # val_pred=torch.reshape(val_pred,(config.batch_size, 1, 7, 5000))
        loss=criterion(val_pred,y)
        val_loss += loss
      val_loss /= len(val_loader)  
      wandb.log({"train_loss": train_loss, 
                 "val_loss": val_loss,
                 "Epoch":epoch})
      

    if (epoch) % 100==0:
      df_input,df_output=get_pred(test_dataset,model)
      model.to(device)
      #plotting the ECG and creating the combined DF
      combined_df=plotECG(df_input,df_output,path=str(ecg_dir))
      #saving combined DF as table on wandB
      input_prediction_table = wandb.Table(dataframe=combined_df)
      ecg_dir_file=ecg_dir.joinpath("ecg.png")
      wandb.log({"ECG": wandb.Image(str(ecg_dir_file))})
      wandb.log({"Input and predictions": input_prediction_table}) 
      
    if (epoch) % 100==0:
      print("one")
      model_dir_model=model_dir.joinpath("model")
      torch.save(model.state_dict(),str(model_dir_model))
      print("two")
      wandb.log_artifact(str(model_dir_model), name='Model', type='Model') 

# Build, train and analyze the model with the pipeline
model = model_pipeline(config)


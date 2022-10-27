"""
Here, we define the autoencoder model.This model is taken from "https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py"
"""
import torch
from torch import nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn import ConvTranspose1d

class ECG_AE_v1(nn.Module):
    def __init__(self,step1=128,step2=64,step3=20):
        super(ECG_AE_v1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000,step1),
            nn.ReLU(),
            nn.Linear(step1,step2),
            nn.ReLU(),
            nn.Linear(step2,step3),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(

            nn.Linear(step3,step2),
            nn.ReLU(),
            nn.Linear(step2,step1),
            nn.ReLU(),
            nn.Linear(step1,35000),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ECG_stacked_AE(nn.Module):
    def __init__(self,step1=128,step2=64,step3=20):
        super(ECG_AE_v1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000,step1),
            nn.ReLU(),
            nn.Linear(step1,step2),
            nn.ReLU(),
            nn.Linear(step2,step3),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(

            nn.Linear(step3,step2),
            nn.ReLU(),
            nn.Linear(step2,step1),
            nn.ReLU(),
            nn.Linear(step1,5000),
        )
        self.decoder2 = nn.Sequential(

            nn.Linear(step3,step2),
            nn.ReLU(),
            nn.Linear(step2,step1),
            nn.ReLU(),
            nn.Linear(step1,5000),
        )
        self.decoder3 = nn.Sequential(

            nn.Linear(step3,step2),
            nn.ReLU(),
            nn.Linear(step2,step1),
            nn.ReLU(),
            nn.Linear(step1,5000),
        )
        self.decoder4 = nn.Sequential(

            nn.Linear(step3,step2),
            nn.ReLU(),
            nn.Linear(step2,step1),
            nn.ReLU(),
            nn.Linear(step1,5000),
        )
        self.decoder5 = nn.Sequential(

            nn.Linear(step3,step2),
            nn.ReLU(),
            nn.Linear(step2,step1),
            nn.ReLU(),
            nn.Linear(step1,5000),
        )
        self.decoder6 = nn.Sequential(

            nn.Linear(step3,step2),
            nn.ReLU(),
            nn.Linear(step2,step1),
            nn.ReLU(),
            nn.Linear(step1,5000),
        )
        self.decoder7 = nn.Sequential(

            nn.Linear(step3,step2),
            nn.ReLU(),
            nn.Linear(step2,step1),
            nn.ReLU(),
            nn.Linear(step1,5000),
        )



    def forward(self, x):
        x = self.encoder(x)
        x1 = self.decoder1(x)
        x2 = self.decoder2(x)
        x3 = self.decoder3(x)
        x4 = self.decoder4(x)
        x5 = self.decoder5(x)
        x6 = self.decoder6(x)
        x7 = self.decoder7(x)
        x_cat=torch.stack([x1,x2,x3,x4,x5,x6,x7])
        #permute for training and validation while training with batches
        if x.ndim == 3:
            #print(f"input has shape of{x.shape} and dimension of {x.ndim},reshaping output to (batchsize,1,7,5000)) ")
            x_cat=torch.permute(x_cat,(1,2,0,3))
        #permute for taking predictions from dataset
        if x.ndim == 2:
            #print(f"input has shape of{x.shape} and dimension of {x.ndim},reshaping output to (1,7,5000)) ")
            x_cat=torch.permute(x_cat,(1,0,2))


        return x_cat


class ECG_AE_conv_leak(nn.Module):
    def __init__(self,step1=128,step2=64,step3=20):
        super(ECG_AE_conv_leak, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,1,101,padding=50),
            nn.Linear(5000,step1),
            nn.LeakyReLU(),
            nn.Linear(step1,step2),
            nn.LeakyReLU(),
            nn.Linear(step2,step3),
            nn.LeakyReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(step3,step2),
            nn.LeakyReLU(),
            nn.Linear(step2,step1),
            nn.LeakyReLU(),
            nn.Linear(step1,35000)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x=torch.reshape(x,( 1, 7, 5000))

        return x

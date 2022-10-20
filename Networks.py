"""
Here, we define the autoencoder model.This model is taken from "https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py"
"""
import torch
from torch import nn

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

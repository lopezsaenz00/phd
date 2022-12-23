import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
from .module import ConvLayer,LinearLayer,ResizeLayerUp,ResizeLayerDown,FlatenLayer,SingleEncoder,SingleDecoder
from .module import Encoder,Decoder
from .module import Encoder2,Decoder2
from .module import Encoder4,Decoder4

class Model(nn.Module):
    def __init__(self,c_in=80,seg_len=30,hid=64):
        super().__init__()
        self.enc = Encoder(c_in,hid=hid)
        self.dec = Decoder(c_out = c_in,c_in=hid)
        self.dense_mean = nn.Linear(hid,hid)
        self.dense_var = nn.Linear(hid,hid) 
    def forward(self,x):
        lat = self.enc(x)
        mean = self.dense_mean(lat.permute(0,2,1)).permute(0,2,1)
        var = self.dense_var(lat.permute(0,2,1)).permute(0,2,1)
        epsilon = torch.normal(torch.zeros(lat.size()),torch.ones(lat.size()))
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        z = mean + torch.exp(0.5*var) * epsilon
        kl = torch.mean(-0.5* torch.sum(1 + var - mean**2 - torch.exp(var), dim=[1,2]))
        out = self.dec(z)
        return out, kl

    def encode(self,x):
        lat = self.enc(x)
        return lat
class SingleModel(nn.Module):
    def __init__(self,c_in=80,factor=3,hid=64,seg_len=30):
        super().__init__()
        self.enc = SingleEncoder(c_in,hid=hid,seg_len=seg_len)
        self.dec = SingleDecoder(c_in = hid, c_out = c_in, seg_len=seg_len)
        self.dense_mean = nn.Linear(hid,hid)
        self.dense_var = nn.Linear(hid,hid) 
    def forward(self,x):
        out = self.enc(x)
        mean = self.dense_mean(out.permute(0,2,1)).permute(0,2,1)
        var = self.dense_var(out.permute(0,2,1)).permute(0,2,1)
        epsilon = torch.normal(torch.zeros(out.size()),torch.ones(out.size()))
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        z = mean + torch.exp(0.5 * var) * epsilon
        kl = torch.mean(-0.5 * torch.sum(1 + var - mean**2 - torch.exp(var),dim=[1,2]))
        out = self.dec(z)
        return out,kl

    def encode(self,x):
        out = self.enc(x)
        mean = self.dense_mean(out.permute(0,2,1)).permute(0,2,1)
        var = self.dense_var(out.permute(0,2,1)).permute(0,2,1)
        epsilon = torch.normal(torch.zeros(out.size()),torch.ones(out.size()))
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        z = mean + torch.exp(0.5 * var) * epsilon
        return z

"""
Created on Fri December 04 15:46:00 2020

@author: Jose Antonio Lopez @ The University of Sheffield
Modules for the FFWD Assessor decision model.

"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
#from .module import ConvLayer,LinearLayer,ResizeLayerUp,ResizeLayerDown,FlatenLayer,SingleEncoder,SingleDecoder
        
        
   
class LinearLayer(nn.Module):
    def __init__(self,c_in,c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.layer = nn.Linear(c_in,c_out)
        self.norm = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)
    def forward(self,x):
        #B,F,T = x.size()
        #x = x.view(B*T,F)
        out = self.layer(x)
        out = self.norm(out)
        #out = out.view(B,self.c_out,T)
        out = self.act(out)
        out = self.drop(out)
        return out
        
class FFWD1(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        
        self.net = nn.Sequential()
        self.net.add_module('linear1',LinearLayer(in_dim,512))
        self.net.add_module('linear2',LinearLayer(512,256))
        self.out_layer = nn.Linear(256,num_cls)
        
    def forward(self,x):
        out = self.net(x)
        out = self.out_layer(out)
        return out
        
    def encode(self,x): 
        return self.net(x)
        
class FFWD2(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        
        self.net = nn.Sequential()
        self.net.add_module('linear1',LinearLayer(in_dim,256))
        self.net.add_module('linear2',LinearLayer(256,128))
        self.out_layer = nn.Linear(128,num_cls)
        
    def forward(self,x):
        out = self.net(x)
        out = self.out_layer(out)
        return out
        
    def encode(self,x): 
        return self.net(x)
        
class FFWD3(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        
        self.net = nn.Sequential()
        self.net.add_module('linear1',LinearLayer(in_dim,512))
        self.out_layer = nn.Linear(512,num_cls)
        
    def forward(self,x):
        out = self.net(x)
        out = self.out_layer(out)
        return out
        
    def encode(self,x): 
        return self.net(x)
        
class FFWD4(nn.Module):
    def __init__(self, in_dim, num_cls):
        super().__init__()
        
        self.net = nn.Sequential()
        self.net.add_module('linear1',LinearLayer(in_dim,256))
        self.out_layer = nn.Linear(256,num_cls)
        
    def forward(self,x):
        out = self.net(x)
        out = self.out_layer(out)
        return out
        
    def encode(self,x): 
        return self.net(x)

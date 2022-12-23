"""

@author: Jose Antonio Lopez @ The University of Sheffield

Torch modules for the transformer network for the assessor model

"""
import copy
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from .attention_network import Predictor
from torch.nn.modules.module import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm
from .activation_MOD import MultiheadAttention
from .speech_conv import ConvBNReLU
import torch.nn.functional as F
import torch.nn.parameter #as Parameter
import math
import sys

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


#positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, pos_input, d_model, dropout = 0.1, max_len = 5000, pos_linear = 0, pos_cnn = 0, cnn_params = None, encoding='pos'):
        """
        Args:
            d_model: int, (encoding) dimension of the model
            max_len: int, maximum normalizatio length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.pad = False
        self.pos_cnn = pos_cnn
        self.pos_linear = pos_linear
        self.d_model = d_model
        self.pos_input = pos_input
        self.max_len = max_len
        self.encoding = encoding
        
        if self.pos_cnn:
              self.pos_conv = ConvBNReLU(out_channels = cnn_params['out_channels'], kernel_sizes = cnn_params['kernel_sizes'], strides = cnn_params['strides'], in_channels=1, padding = cnn_params['padding'])
              # as I cannot always predict the final dimensionality, just send a forward pass to see the output dimentionality
              self.pos_conv.eval()
              x, x_len = self.pos_conv(torch.Tensor(2,self.max_len, self.pos_input), src_lengths = self.max_len)
              self.linear_input = x.shape[-1]
              self.max_len = x.shape[-2]
        else:
              self.linear_input = self.pos_input
            
        if self.pos_cnn == 0 and self.pos_linear == 0:
            self.pad = True
            self.pos_input = self.d_model
            print("Positional embedding uses pad = True if needed.") 
            
        if self.pos_linear:
              self.linear_encoder = Predictor(self.d_model, self.linear_input, self.d_model,self.pos_linear)
              

        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1,self.max_len, self.d_model)
        #by not selection encoding = 'pos' , we just do not implement sin-cos positional encoding
        if self.encoding == 'pos':
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, src_lengths = 0):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        
        if self.pos_cnn:
            x, x_len = self.pos_conv(x, src_lengths = src_lengths)
        
        if self.pos_linear:
            x = self.linear_encoder(x)
        
        if  self.pad and x.shape[-1]<self.pe.shape[-1]:
            zx = torch.zeros(x.shape[0],self.max_len, self.d_model)
            if use_cuda:
                zx = zx.cuda()
            zx[:,:x.size(1), :x.size(2)] += x
            return self.dropout(zx + self.pe) 
            
        else:
            #x = x + self.pe[:x.size(0)]
            return self.dropout(x +  self.pe )
    
        
class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']   #check what this one does

    def __init__(self, pos_encoder, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.pos_encoder = pos_encoder
        self.init_weights()
        
    def init_weights(self):
    
        if self.pos_encoder.pos_cnn:
            for m in self.pos_encoder.pos_conv.convolutions:
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        for m in self.layers:
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
                
        

    def forward(self, src, mask= None, src_key_padding_mask= None, need_weights=True, avg_weights=False ):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        src = self.pos_encoder(src)
        src = torch.transpose(src,0,1)      # to swap the batch dimension and position dimension to match the transformer layer format.
        attn_list = []
        if need_weights:
            for mod in self.layers:
                src, attn_w = mod(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask, need_weights=need_weights, avg_weights=avg_weights )
                attn_list.append(attn_w)
            return src, torch.stack(attn_list)
        else:
            for mod in self.layers:
                src = mod(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask, need_weights=need_weights, avg_weights=avg_weights )
            return src
                
        # if self.norm is not None:
            # output = self.norm(output)


        
        
        
class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None, need_weights=True, avg_weights=False ):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            #use (lenght, batch size, emb_dim) format for src

        Shape:
            see the docs in Transformer class.
        """
        # src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              # key_padding_mask=src_key_padding_mask)[0]
        
        src2, attn_w = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask,  need_weights=need_weights, avg_weights=avg_weights, attn_mask=src_mask)
                                                     
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if need_weights:
            return src, attn_w
        else:
            return src
        
        
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
        
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

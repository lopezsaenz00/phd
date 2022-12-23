"""

@author: Jose Antonio Lopez @ The University of Sheffield

Torch modules for the lstm+attn network

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter #as Parameter
import math
from .mult_head_attn_mod import MultiheadAttention_MOD as MultiheadAttention
import sys

# Encoder
class LstmNet(nn.Module):
	"""
	input:  features (B * N * D), B=batch_size, N=length_of_utt, D=dimension
	hidden: layers (num_layers, 2) of size (B * N * H), H=hidden_size(512)
	output: lstm hidden states (B * N * O), O=outlayer_size(1024)
	"""
	def __init__(self,input_size,hidden_size,num_layers, bidirectional=True, dropout = 0):
		super(LstmNet, self).__init__()
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers,bidirectional=bidirectional, dropout = dropout)
		
	def forward(self,x):
		x = torch.transpose(x,0,1)      # to swap the batch dimension and position dimension
		outputs, _ = self.lstm(x)
		# output is shape seq_len, batch, num_directions * hidden_size)
		return outputs

#CNN network
class CNN(nn.Module):
	"""
	input:  o_channels : int or list of ints, for every layer

	"""
	def __init__(self, o_channels, in_channel = 1, k_size = 3, stride = 1):
		super(CNN, self).__init__()
		if not( isinstance(o_channels, list) ): 
			o_channels = [o_channels]
            
		in_channels = [in_channel] + o_channels
            
		modules = []
		modules.append( nn.Conv2d(in_channels = in_channels[0] , out_channels = o_channels[0], kernel_size = k_size, stride = stride) )
		modules.append( nn.BatchNorm2d(o_channels[0]) )
		modules.append( nn.ReLU() )
		modules.append( nn.MaxPool2d( kernel_size=3, stride=2) )
		if len( o_channels ) > 1:
			for i,j in enumerate(o_channels[1:]):
				modules.append( nn.Conv2d(in_channels = in_channels[1+i] , out_channels = j, kernel_size = k_size, stride = stride) )
				modules.append( nn.BatchNorm2d(j) )
				modules.append( nn.ReLU() )
				modules.append( nn.MaxPool2d( kernel_size=3, stride=2) )
                
		self.cnn = nn.Sequential(*modules)
        
	def forward(self, x):
		output = self.cnn(x)
		#outputs the flat info, 1 x (channel, h, q).flatten
		return output.view(output.size(0), -1)
        
        
        
# Global Attention Uses the average across time per dimension 
class CNNPredictor(nn.Module):
	def __init__(self, o_channels, in_channel = 1, k_size = 3, stride = 1, hidden_size = 256, num_layers = 2, nclasses = 1, w=49, h =39):
		super(CNNPredictor, self).__init__()
		self.cnn = CNN(o_channels = o_channels, in_channel = in_channel, k_size = k_size, stride = stride )
		#determine the required layer size in a lzy way...
		fc_layer_dim = torch.rand((2,1,w,h))
		fc_layer_dim = self.cnn(fc_layer_dim ).shape[1]
		self.fc = Predictor(nclasses,fc_layer_dim, hidden_size, num_layers)
		del fc_layer_dim

	def forward(self, x):
		cn = self.cnn(x.unsqueeze(1))
		return self.fc(cn)
        

# Global Attention Uses the average across time per dimension 
class Attention(nn.Module):
	"""
	input:  lstm hidden states (L * B * O), O=outlayer_size(1024)
	att:    dual attention network layer (L * B * AH), AH=attention_hidden_size(128)
	output: context vector (B * DH), DH=dan_hidden_size(1024)
	"""
	def __init__(self, dan_hidden_size, attention_hidden_size):
		super(Attention, self).__init__()
		N = dan_hidden_size		#N
		N2 = attention_hidden_size	#AH

		self.W = nn.Linear(N,N2)		# input size N, output size N2
		self.W_m = nn.Linear(N,N2)
		self.W_h = nn.Linear(N2,1)	#should be scalar

	def forward(self, hyp, dan_hidden_size, attention_hidden_size, BATCHSIZE):
		N = dan_hidden_size
		N2 = attention_hidden_size

		#print("hyp=", hyp.shape)
		m = hyp.mean(0).unsqueeze(0)
		#print("m=", m.shape)
		m = m.permute(1,0,2)
		#print("m permute=", m.shape)
		hyp = hyp.permute(1,0,2)
		#print("hyp permute=", hyp.shape)
		mx = m.repeat(1, hyp.size(1),1)
		#print("mx=", mx.shape)
		h = torch.tanh(self.W(hyp))*torch.tanh(self.W_m(mx))
		#print("h=", h.shape)
		a = F.softmax(self.W_h(h),dim=1)
		#print("a=", a.shape)
		c = (a.repeat(1,1,N)*hyp).sum(1)
		#print("c=", c.shape)
		#sys.exit()
		return c



# final layer for classifying emotion
class Predictor(nn.Module):
	"""
	input:  context vector (B * DH), DH=dan_hidden_size(1024)
	output: prediction (B * NE), NE=num_emotions(6) 
	"""
	def __init__(self,num_emotions,input_size, hidden_size, num_layers, dropp = None):#,output_scale_factor = 1, output_shift = 0):
		super(Predictor, self).__init__()
		modules = []
		
		if num_layers:
        
			if not( isinstance(hidden_size, list) ): 
				hidden_size = [hidden_size] * num_layers
        
			for layer in range( num_layers):
				if layer == 0:
					modules.append(nn.Linear(input_size, hidden_size[0]))
					self.init_weights(modules[-1])
				else:
					modules.append(nn.Linear(hidden_size[layer-1], hidden_size[layer]))
					self.init_weights(modules[-1])
					if dropp:
						modules.append(nn.Dropout( p = dropp))
				
				modules.append(nn.ReLU())
			
			modules.append(nn.Linear(hidden_size[layer], num_emotions))
			self.init_weights(modules[-1])
				
		else:
			modules.append(nn.Linear(input_size, num_emotions))
		
		self.net = nn.Sequential(*modules)

	def forward(self,x):
		return self.net.forward(x)
		
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)
			torch.nn.init.zeros_(m.bias)


class Norm_Layer(nn.Module):
	""" Add and Norm Layer
    
    param: norm_len - flattened sequence vectors ( FRAME*DIMENSIONS )
    parm: p - dropout probability
    
    """
    
	def __init__(self, input_size, p = 0.1):
		super(Norm_Layer, self).__init__()
		self.layer_norm = nn.LayerNorm(input_size)
		self.droppy = nn.Dropout( p = p)
        
	def forward(self, x ):
		x = self.layer_norm( x )
		return self.droppy(x)
        
class BNorm_Layer(nn.Module):
	""" Add and batch Norm Layer
    
    param: norm_len - flattened sequence vectors ( FRAME*DIMENSIONS )
    parm: p - dropout probability
    
    """
    
	def __init__(self, input_size, p = 0.1):
		super(BNorm_Layer, self).__init__()
		self.bnorm = nn.BatchNorm1d(input_size)
		self.droppy = nn.Dropout( p = p)
        
	def forward(self, x ):
		x = self.bnorm( x )
		return self.droppy(x)
        
class Add_N_Norm(nn.Module):
	""" Add and Norm Layer
    input:  lstm hidden states (FRAME x BATCH x DIMENSIONS)
    input: Bahd context vector ( BATCH X FRAME X DIMENSIONS)
    
    param: norm_len - flattened sequence vectors ( FRAME*DIMENSIONS )
    parm: p - dropout probability
    
    """
    
	def __init__(self, norm_len, p = 0.1, flat = True ):
		super(Add_N_Norm, self).__init__()
		self.layer_norm = nn.LayerNorm(norm_len)
		self.droppy = nn.Dropout( p = p)
		self.flat = flat
        
	def forward(self, hyp, context ):
		hyp = hyp.permute(1,0,2) #now -> batch, frame, emb_dim
		if self.flat:
			add_ = torch.flatten(hyp + context, start_dim = 1)
		else:
			add_ = hyp + context
		emb = self.layer_norm( add_ )
		return self.droppy(emb)
        
        
class BahdanauAttention(nn.Module):
	""" Bahdanau Attention."""
	def __init__(self, query_dim, value_dim, embed_dim, soft_dim = 2, normalize=False):
		super(BahdanauAttention, self).__init__()
		self.embed_dim = embed_dim
		self.query_dim = query_dim
		self.value_dim = value_dim
		self.query_proj = nn.Linear(self.query_dim, embed_dim, bias=False) #source?
		self.value_proj = nn.Linear(self.value_dim, embed_dim, bias=False) #target?
		self.v = torch.nn.parameter.Parameter(torch.Tensor(embed_dim))
		self.normalize = normalize
		if self.normalize:
			self.b = torch.nn.parameter.Parameter(torch.Tensor(embed_dim))
			self.g = torch.nn.parameter.Parameter(torch.Tensor(1))
		self.reset_parameters()
		self.soft_dim = soft_dim
		if self.soft_dim == 1:
			print("Softmax normalization across FRAMES")
		elif self.soft_dim == 2:
			print("Softmax normalization across DIMENSIONS")
		elif self.soft_dim == -1:   
			print("2DSoftmax normalization on the Attention Module")
                
	def reset_parameters(self):
		self.query_proj.weight.data.uniform_(-0.1, 0.1)
		self.value_proj.weight.data.uniform_(-0.1, 0.1)
		nn.init.uniform_(self.v, -0.1, 0.1)
		if self.normalize:
			nn.init.constant_(self.b, 0.0)
			nn.init.constant_(self.g, math.sqrt(1.0 / self.embed_dim))
        
	def forward(self, query, value ):      
		query = query.permute(1,0,2)
		value = value.permute(1,0,2)
		projected_query = self.query_proj(query)
		key = self.value_proj(value)
        
		if self.normalize:
			# normed_v = g * v / ||v||
			normed_v = self.g * self.v / torch.norm(self.v)
			attn_scores = normed_v * torch.tanh(projected_query + key + self.b) 
            
		else:
			attn_scores = self.v * torch.tanh(projected_query + key)

		if self.soft_dim == -1:
			attn_scores = attn_scores.unsqueeze(1)
			attn_scores = nn.Softmax(2)(attn_scores.view(*attn_scores.size()[:2], -1)).view_as(attn_scores)
			attn_scores = attn_scores.squeeze(1)
            
		else:
			attn_scores = F.softmax(attn_scores, dim = self.soft_dim)
            
		context = attn_scores * value
        
		return context, attn_scores
        
        
class MultiHeadSelfAttention(nn.Module):
	""" MultiHead Self Attention."""
	def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None ):
		super(MultiHeadSelfAttention, self).__init__()
		self.multihead = MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        
	def forward(self, value, key_padding_mask=None,need_weights=True, avg_weights=True, attn_mask=None):
		#use (lenght, batch size, emb_dim) format
		context, attn_w = self.multihead(value, value, value, key_padding_mask=key_padding_mask,need_weights=need_weights, avg_weights=avg_weights, attn_mask=attn_mask)
		context = context.permute(1,0,2) #now -> batch, frame, dim
		return context, attn_w
         
        
class HierarMultiHeadSelfAttention(nn.Module):
	""" Hierarchical MultiHead Self Attention."""
	""" Only for 2 layers."""
	def __init__(self, norm_len, embed_dim, num_heads, avg_weights=True, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
		super(HierarMultiHeadSelfAttention, self).__init__()
		self.multihead1 = MultiHeadSelfAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
		self.add_n_norm = Add_N_Norm( norm_len, p = 0.1, flat = False)
		self.multihead2 = MultiHeadSelfAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        
	def forward(self, value, key_padding_mask=None,need_weights=True, avg_weights=True, attn_mask=None):
		#use (lenght, batch size, emb_dim) format
		context1, attn_w1 = self.multihead1(value, key_padding_mask=key_padding_mask,need_weights=need_weights, avg_weights=avg_weights, attn_mask=attn_mask) # value is assummed to be (length, batch size, emb_dim) format
		#context1 is -> (batch, length, dim)
		#value is changed to (batch, length, dim) in self.add_n_norm
		value2 = self.add_n_norm(value, context1) # value2 is -> (batch, frame, dim), needs to change
		value2 = value2.permute(1,0,2) #now -> lenght, batch size, dim
		context2, attn_w2 = self.multihead2(value2, key_padding_mask=key_padding_mask,need_weights=need_weights, avg_weights=avg_weights, attn_mask=attn_mask)
		return context1, context2, attn_w1, attn_w2 #the context output  -> batch, frame, dim
        
        
class TransformerEncoder(nn.Module): 
	""" Transformer Encoder - Multihead attention."""
	def __init__(self, input_size, nhead, dim_feedforward, num_layers, norm = False ):
		super(TransformerEncoder, self).__init__()
		if norm:
			layer_norm = nn.LayerNorm(input_size)
		else:
			layer_norm = None
        
		encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=dim_feedforward, batch_first = True)
		self.trans_enc = nn.TransformerEncoder(encoder_layer, num_layers, norm=self.layer_norm)
        
	def forward(self, x): 
		return self.trans_enc(x)
        
        
#class LuongGlobalAttention(nn.Module):
#	""" Luong Attention."""
#	def __init__(self, query_dim, value_dim,  score = 'dot' , scale=True):
#		super(LuongGlobalAttention, self).__init__()
#		self.value_dim = value_dim
#		self.query_dim = query_dim
#		self.value_proj = nn.Linear(self.value_dim, self.query_dim, bias=False)
#		self.scale = scale
#		self.score = score
#		if self.scale:
#			self.g = torch.nn.parameter.Parameter(torch.Tensor(1))
#		self.reset_parameters()
#        
#	def reset_parameters(self):
#		self.value_proj.weight.data.uniform_(-0.1, 0.1)
#		if self.scale:
#			nn.init.constant_(self.g, 1.0)
#            
#	def forward(self, h_s):
#	#Notation from Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025.
#		print("h_s")
#		print(h_s.shape)
#		h_s = h_s.permute(1,0,2)
#		print("h_s permute")
#		print(h_s.shape)
#        
#        h_t = h_s[:, -1, :]
#        print("h_t")
#        print(h_t.shape)
#		sys.exit()
    
    
        
#ASIF'S ATTENTION CODE
# class BahdanauAttention(nn.Module):
    # """ Bahdanau Attention."""
    # def __init__(self, query_dim, value_dim, embed_dim, normalize=True):
        # super(BahdanauAttention, self).__init__()
        # #super().__init__(query_dim, value_dim, embed_dim)
        # self.embed_dim = embed_dim
        # self.query_dim = query_dim
        # self.value_dim = value_dim
        # self.query_proj = nn.Linear(self.query_dim, embed_dim, bias=False) #source?
        # self.value_proj = nn.Linear(self.value_dim, embed_dim, bias=False) #target?
        # self.v = torch.nn.parameter.Parameter(torch.Tensor(embed_dim))
        # self.normalize = normalize
        # if self.normalize:
            # self.b = torch.nn.parameter.Parameter(torch.Tensor(embed_dim))
            # self.g = torch.nn.parameter.Parameter(torch.Tensor(1))
        # self.reset_parameters()
    # def reset_parameters(self):
        # self.query_proj.weight.data.uniform_(-0.1, 0.1)
        # self.value_proj.weight.data.uniform_(-0.1, 0.1)
        # nn.init.uniform_(self.v, -0.1, 0.1)
        # if self.normalize:
            # nn.init.constant_(self.b, 0.0)
            # nn.init.constant_(self.g, math.sqrt(1.0 / self.embed_dim))
    # def forward(self, query, value, key_padding_mask=None, state=None):
        # ###debug
        # print("query shape")
        # print(query.shape)
        # print("value shape")
        # print(value.shape)
        # print("key_padding_mask")
        # print(key_padding_mask)
        # print("key_padding_mask is not None")
        # print(key_padding_mask is not None)
        # ###debug
        # # projected_query: 1 x bsz x embed_dim
        # projected_query = self.query_proj(query).unsqueeze(0)
        # key = self.value_proj(value)  # len x bsz x embed_dim
        
        # ###debug
        # print("projected_query shape with unsqueeze")
        # print(projected_query.shape)
        # print("key shape ")
        # print(key.shape)
        # ###debug
        
        # if self.normalize:
            # # normed_v = g * v / ||v||
            # normed_v = self.g * self.v / torch.norm(self.v)
            # attn_scores = (  normed_v * torch.tanh(projected_query + key + self.b)  ).sum(dim=2)  # len x bsz
        # else:
            # attn_scores = self.v * torch.tanh(projected_query + key).sum(dim=2)
        # if key_padding_mask is not None:
            # attn_scores = (    attn_scores.float().masked_fill_(key_padding_mask, float("-inf")).type_as(attn_scores) )  # FP16 support: cast to float and back
        # attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz
        # # sum weighted value. context: bsz x value_dim
        # context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        # next_state = attn_scores
        # return context, attn_scores, next_state
        
# class LuongAttention(nn.Module):
    # """ Luong Attention."""
    # def __init__(self, query_dim, value_dim,  scale=True):
        # super(LuongAttention, self).__init__()
        # #super().__init__(query_dim, value_dim, embed_dim)
        # self.value_dim = value_dim
        # self.query_dim = query_dim
        # self.value_proj = nn.Linear(self.value_dim, self.query_dim, bias=False)
        # self.scale = scale
        # if self.scale:
            # self.g = torch.nn.parameter.Parameter(torch.Tensor(1))
        # self.reset_parameters()
    # def reset_parameters(self):
        # self.value_proj.weight.data.uniform_(-0.1, 0.1)
        # if self.scale:
            # nn.init.constant_(self.g, 1.0)
    # def forward(self, query, value, key_padding_mask=None, state=None):
        # query = query.unsqueeze(1)  # bsz x 1 x query_dim
        
        # ###debug
        # print("this is query unsqueeze(1)" )
        # print(query.shape)
        # ###debug
        
        # key = self.value_proj(value).transpose(0, 1)  # bsz x len x query_dim
        # ###debug
        # print("this is key (value proj)" )
        # print(key.shape)
        # ###debug
        
        # attn_scores = torch.bmm(query, key.transpose(1, 2)).squeeze(1)
        # attn_scores = attn_scores.transpose(0, 1)  # len x bsz
        # if self.scale:
            # attn_scores = self.g * attn_scores
        # if key_padding_mask is not None:
            # attn_scores = (
                # attn_scores.float()
                # .masked_fill_(key_padding_mask, float("-inf"))
                # .type_as(attn_scores)
            # )  # FP16 support: cast to float and back
        # attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz
        # # sum weighted value. context: bsz x value_dim
        # context = (attn_scores.unsqueeze(2) * value).sum(dim=0)
        # next_state = attn_scores
        # return context, attn_scores, next_state
        
        
## final layer for classifying emotion
#class Predictor(nn.Module):
#	"""
#	input:  context vector (B * DH), DH=dan_hidden_size(1024)
#	output: prediction (B * NE), NE=num_emotions(6) 
#	"""
#	def __init__(self,num_emotions,input_size, hidden_size, num_layers):#,output_scale_factor = 1, output_shift = 0):
#		super(Predictor, self).__init__()
#		self.linears = []
#		if num_layers > 1:
#			for i in range(num_layers-1):
#				if i == 0:
#					self.linears.append(nn.Linear(input_size, hidden_size))
#				else:
#					self.linears.append(nn.Linear(hidden_size, hidden_size))
#						
#			self.fc = nn.Linear(hidden_size, num_emotions)	
#		else:
#			self.fc = nn.Linear(input_size, num_emotions)

#	def forward(self,x):
#		if self.linears != []:
#			for i in range(len(self.linears)):
#				x = self.linears[i](x)
#				x = F.relu(x)
#	
#		x = self.fc(x)
#		return x


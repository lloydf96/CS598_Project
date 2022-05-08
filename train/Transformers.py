
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torch
import pandas as pd

class LF_Transformer(torch.nn.Module):
    def __init__(self,vital_dim,inter_dim,threshold = 0.5):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.vital_encoder = Transformer(vital_dim)
        self.inter_encoder = Transformer(inter_dim)
        self.fc1 = nn.Linear(256,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,2)
        self.vital_dim = vital_dim
        self.inter_dim = inter_dim
        
    def forward(self,pos,neg):
        (inter,vital,inter_mask,vital_mask) = pos
        (inter_neg,vital_neg,inter_neg_mask,vital_neg_mask) = neg
        
        seq_len = inter.shape[1]
        inter = self.inter_encoder(inter,inter_mask)
        vital = self.vital_encoder(vital,vital_mask)
        c = torch.cat([inter,vital],dim = 1)
        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)
        return c
    
    
class Transformer(torch.nn.Module):
    def __init__(self,dim):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.dim = dim
        self.embed = nn.Linear(dim,128)
        self.relu = nn.ReLU()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
    def forward(self,x,mask):
        
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        x = x.reshape(-1,self.dim)
        
        x = self.embed(x)
        x = self.relu(x)      
        x = x.reshape(-1,seq_len,128)
        x = x.permute([1,0,2])
        mask_tensor = (torch.arange(seq_len).repeat([batch_size,1])).to('cuda') > torch.unsqueeze(mask,dim = 1)
        output = self.transformer_encoder(x,src_key_padding_mask = mask_tensor).permute([1,0,2])
        output = torch.sum(output,dim = 1)
        
        return output
    

class LF_pool_Transformer(torch.nn.Module):
    def __init__(self,vital_dim,inter_dim,threshold = 0.5):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.vital_encoder = Transformer(vital_dim)
        self.inter_encoder = Transformer(inter_dim)
        self.fc1 = nn.Linear(128,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,2)
        self.vital_dim = vital_dim
        self.inter_dim = inter_dim
        
    def forward(self,pos,neg):
        (inter,vital,inter_mask,vital_mask) = pos
        (inter_neg,vital_neg,inter_neg_mask,vital_neg_mask) = neg
        
        seq_len = inter.shape[1]
        inter = self.inter_encoder(inter,inter_mask)
        vital = self.vital_encoder(vital,vital_mask)

        c = torch.cat([inter[:,None],vital[:,None]],dim = 1)
        c = torch.max(c,dim = 1)[0]
        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)

        return c
    
class EFI_Transformer(torch.nn.Module):
    def __init__(self,vital_dim,inter_dim,threshold = 0.5):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        
        self.vital_dim = vital_dim
        self.inter_dim = inter_dim
        self.embed = nn.Linear(vital_dim+inter_dim,256)
        self.encoder = vanilla_Transformer()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,2)
        
    def forward(self,pos,neg):
        (inter,vital,inter_mask,vital_mask) = pos
        (inter_neg,vital_neg,inter_neg_mask,vital_neg_mask) = neg
        inter_vital = torch.cat([inter,vital],dim = 2)
        seq_len = inter_vital.shape[1]
        inter_vital = inter_vital.reshape(-1,self.inter_dim + self.vital_dim)
        inter_vital = self.embed(inter_vital)
        inter_vital = self.relu(inter_vital)      
        inter_vital = inter_vital.reshape(-1,seq_len,256)
        
#         print(torch.cat([inter_mask[:,None],vital_mask[:,None]],axis = 1).shape)
        mask = torch.min(torch.cat([inter_mask[:,None],vital_mask[:,None]],dim = 1),dim = 1)[0]
#         print(mask.shape)
        c= self.encoder(inter_vital,mask)
        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)
        return c
    
class EFE_Transformer(torch.nn.Module):
    def __init__(self,vital_dim,inter_dim,threshold = 0.5):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.encoder = vanilla_Transformer()
        self.vital_dim = vital_dim
        self.inter_dim = inter_dim
        self.vital_embed = nn.Linear(vital_dim,128)
        self.inter_embed = nn.Linear(inter_dim,128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,2)

        
    def forward(self,pos,neg):
        (inter,vital,inter_mask,vital_mask) = pos
        (inter_neg,vital_neg,inter_neg_mask,vital_neg_mask) = neg
        
        seq_len = inter.shape[1]
        inter = inter.reshape(-1,self.inter_dim)
        inter = self.inter_embed(inter)
        inter = self.relu(inter)      
        inter = inter.reshape(-1,seq_len,128)
        
        seq_len = vital.shape[1]
        vital = vital.reshape(-1,self.vital_dim)
        vital = self.vital_embed(vital)
        vital = self.relu(vital)      
        vital = vital.reshape(-1,seq_len,128)
        
#         print(vital.shape)
#         print(inter.shape)
        
        inter_vital = torch.cat([inter,vital],dim = 2)
        
#         print(inter_mask.shape)
#         print(vital_mask.shape)
        
#         print(torch.cat([inter_mask[:,None],vital_mask[:,None]],axis = 1).shape)
        mask = torch.min(torch.cat([inter_mask[:,None],vital_mask[:,None]],dim = 1),dim = 1)[0]
#         print(mask.shape)
        c = self.encoder(inter_vital,mask)
        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)
        return c


class vanilla_Transformer(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
    def forward(self,x,mask):

        seq_len = x.shape[1]
        batch_size = x.shape[0]
        x = x.permute([1,0,2])
        mask_tensor = (torch.arange(seq_len).repeat([batch_size,1])).to('cuda') > torch.unsqueeze(mask,dim = 1)
        output = self.transformer_encoder(x,src_key_padding_mask = mask_tensor).permute([1,0,2])
        output = torch.sum(output,axis = 1)
        return output
    

class SDPRL_Transformer(torch.nn.Module):
    def __init__(self,vital_dim,inter_dim,threshold = 0.5):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.vital_encoder = Transformer(vital_dim)
        self.inter_encoder = Transformer(inter_dim)
        self.fc1 = nn.Linear(128,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,2)
        self.vital_dim = vital_dim
        self.inter_dim = inter_dim
        
    def forward(self,pos,neg):
        (inter,vital,inter_mask,vital_mask) = pos
        (inter_neg,vital_neg,inter_neg_mask,vital_neg_mask) = neg
        
        if torch.isnan(inter).sum() :
            print('inter_mat')
        if  torch.isnan(vital).sum() :
            print('vital_mat')
        if torch.isnan(inter_neg).sum():
            print('inter_mat_neg')
        if torch.isnan(vital_neg).sum():
            print('vital_mat_neg')
        
        seq_len = inter.shape[1]
        inter = self.inter_encoder(inter,inter_mask)
        vital = self.vital_encoder(vital,vital_mask)
        
        if torch.isnan(inter).sum() :
            print('inter_mat post encoder')
        if  torch.isnan(vital).sum() :
            print('vital_mat post encoder')

#         print(inter.shape)
#         print(vital.shape)
        
#         print(inter_neg.shape)
#         print(vital_neg.shape)
        
        inter_neg = inter_neg.reshape(-1,seq_len,self.inter_dim)
        vital_neg = vital_neg.reshape(-1,seq_len,self.vital_dim)
        inter_neg_mask = inter_neg_mask.reshape(-1)
        vital_neg_mask = vital_neg_mask.reshape(-1)
        
#         print(inter_neg.shape)
#         print(vital_neg.shape)
        
        inter_neg = self.inter_encoder(inter_neg,inter_neg_mask)
        vital_neg = self.vital_encoder(vital_neg,vital_neg_mask)
        
        if torch.isnan(inter_neg).sum():
            print('inter_mat_neg post encoder')
        if torch.isnan(vital_neg).sum():
            print('vital_mat_neg post encoder')
#         print(inter_neg.shape)
#         print(vital_neg.shape)
        
        inter_neg = inter_neg.reshape(-1,32,128)
        vital_neg = vital_neg.reshape(-1,32,128)
        
#         print(inter_neg.shape)
#         print(vital_neg.shape)
#         print(inter.shape)
#         print(vital.shape)

        c = torch.cat([inter[:,None],vital[:,None]],dim = 1)
#         print(c.shape)
        c = torch.max(c,dim = 1)[0]
#         print(c.shape)
        c = self.fc1(c)
        c = self.relu(c)
        c = self.fc2(c)
#         print("iteration_done")
        return c,inter,vital,inter_neg,vital_neg


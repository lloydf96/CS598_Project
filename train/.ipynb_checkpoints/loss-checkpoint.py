from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torch
import pandas as pd

class SDPRL_Loss(nn.Module):
    def __init__(self, tau = 0.1,lam_1 = 0.05,lam_2 = 0.05,lam_c = 1):
        super(SDPRL_Loss, self).__init__()
        self.sim = nn.CosineSimilarity(dim = 2)
        self.CE = nn.CrossEntropyLoss(reduction = 'mean')
        self.tau = tau
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.lam_c = lam_c

    def forward(self,output,y):
        c,inter_pos,vital_pos,inter_neg,vital_neg = output
        #vital_pos : batch_size,n
        #inter_pos : batch_size,n
        #vital_neg : batch_size,32,n
        #inter_neg : batch_size,32,n
        #c : batch_size,2
        #y : batch_size,1
        vital_pos = torch.unsqueeze(vital_pos,dim = 1)
        inter_pos = torch.unsqueeze(inter_pos,dim = 1)
#         print(vital_pos.shape)
#         print(vital_neg.shape)
#         print(inter_pos.shape)
#         print(inter_neg.shape)
        pos = torch.squeeze(torch.exp(self.sim(vital_pos,inter_pos)/self.tau))
#         print(pos.shape)
        neg_m12 = torch.sum(torch.exp(self.sim(vital_pos,inter_neg)/self.tau),dim = 1)
#         print(self.sim(vital_pos,inter_neg).shape)
#         print(neg_m12.shape)
        neg_m21 = torch.sum(torch.exp(self.sim(vital_neg,inter_pos)/self.tau),dim = 1)
#         print(self.sim(vital_neg,inter_pos).shape)
#         print(neg_m21.shape)
#         print(pos,neg_m12,neg_m21)
        if(torch.isnan(neg_m12).sum()):
            print(neg_m12)
            pd.DataFrame(torch.squeeze(vital_pos).detach().to('cpu').numpy()).to_csv('vital_pos.csv')
            pd.DataFrame(torch.squeeze(inter_neg.reshape(-1,256)).detach().to('cpu').numpy()).to_csv('inter_neg.csv')
            
            return None
            
        if(torch.isnan(neg_m21).sum()):
            print(neg_m21)
            pd.DataFrame(torch.squeeze(vital_neg.reshape(-1,256)).to('cpu').detach().numpy()).to_csv('vital_neg.csv')
            pd.DataFrame(torch.squeeze(inter_pos).to('cpu').detach().numpy()).to_csv('inter_pos.csv')
            
            return None
            
        Lm12 = -1*torch.log(pos/neg_m12)
        Lm21 = -1*torch.log(pos/neg_m21)
        L_task = self.CE(c,y)
        
        loss = torch.sum(self.lam_1*Lm12 + self.lam_2*Lm21) + self.lam_c*L_task
#         print(loss)
        if torch.isnan(loss):
            print("loss went NAN")
        return loss
                     
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torch
import pandas as pd

class EHRDataset(Dataset):
    ''' The class is used by dataloader to generate tensors of a given batch size.'''
    def __init__(self,intervention,vital_signs,patients,y,N_neg = 32,task = 'los'):
        
        self.intervention = intervention
        self.vital_signs = vital_signs
        
        self.subjects = self.vital_signs.index.get_level_values(0).drop_duplicates()
        self.subjects = self.subjects[self.subjects.isin(intervention.index.get_level_values(0).drop_duplicates())]
        self.subjects = self.subjects[self.subjects.isin(patients.index.get_level_values(0).drop_duplicates())]
        patients['gender'] = (patients['gender'] == 'M').astype(int)
        self.patients = pd.get_dummies(patients,columns = ['ethnicity','insurance'])
        
        self.vital_signs = self.vital_signs.join(self.patients, on =['subject_id','hadm_id','icustay_id'],how = 'left')
        self.intervention = self.intervention.join(self.patients, on =['subject_id','hadm_id','icustay_id'],how = 'left')
        self.N_neg = N_neg
        
        if task == 'los':
            self.y = y.los_7.astype(int)
        else :
            self.y = y.mort_icu
            
        self.sample_set = pd.DataFrame(data = zip(self.subjects.values,self.y.values),columns = ['subject_id','y'],index = list(range(len(self.subjects))))
           
    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.sample_set.loc[idx,'subject_id']
        intervention = self.intervention.loc[subject_id]
        vital_signs = self.vital_signs.loc[subject_id]
        y = self.y.loc[subject_id]
        samples_to_choose = list(self.sample_set[~(self.sample_set.y == y.item())].subject_id.values)
        neg_list = np.random.choice(samples_to_choose,self.N_neg,replace = False)
        intervention_neg = self.intervention.loc[neg_list]
        
        vital_signs_neg = self.vital_signs.loc[neg_list]
        
        return intervention.values,vital_signs.values,intervention_neg,vital_signs_neg,y.values,self.N_neg
        
def collate_fn(data):
    intervention,vital_signs,intervention_neg,vital_signs_neg,y,N_neg = zip(*data)
    batch_size = len(intervention)
    
    N_neg = N_neg[0]
    max_time = max([inter.shape[0] for inter in intervention] + [inter.reset_index()[['subject_id','hadm_id']].groupby('subject_id').count().max().item() for inter in intervention_neg])
  
    intervention_width = intervention[0].shape[1]
    vital_width = vital_signs[0].shape[1]
    
    vital_mat = torch.zeros(batch_size,max_time,vital_width)
    inter_mat = torch.zeros(batch_size,max_time,intervention_width)
    vital_mat_neg = torch.zeros(batch_size,N_neg,max_time,vital_width)
    inter_mat_neg = torch.zeros(batch_size,N_neg,max_time,intervention_width)
    vital_mask = torch.zeros(batch_size)
    inter_mask = torch.zeros(batch_size)
    vital_neg_mask = torch.zeros(batch_size,N_neg)
    inter_neg_mask = torch.zeros(batch_size,N_neg)

    for i in range(batch_size):
        inter = intervention[i]
        vital = vital_signs[i]
        inter_mat[i,:inter.shape[0]] = torch.FloatTensor(inter)
        vital_mat[i,:vital.shape[0]] = torch.FloatTensor(vital)
        vital_mask[i] = vital.shape[0]-1
        inter_mask[i] = inter.shape[0]-1
        
        subject_id = intervention_neg[i].index.get_level_values(0).drop_duplicates()

        for j,j_idx in enumerate(subject_id):

            inter_neg = intervention_neg[i].loc[j_idx].values
            vital_neg = vital_signs_neg[i].loc[j_idx].values
            inter_mat_neg[i,j,:inter_neg.shape[0]] = torch.FloatTensor(inter_neg)
            vital_mat_neg[i,j,:vital_neg.shape[0]] = torch.FloatTensor(vital_neg)
            vital_neg_mask[i,j] = vital_neg.shape[0]-1
            inter_neg_mask[i,j] = inter_neg.shape[0]-1
        
    
#     if torch.isnan(inter_mat).sum() :
#         print('inter_mat')
#     if  torch.isnan(vital_mat).sum() :
#         print('vital_mat')
#     if torch.isnan(inter_mat_neg).sum():
#         print('inter_mat_neg')
#     if torch.isnan(vital_mat_neg).sum():
#         print('vital_mat_neg')
        
        
    return (inter_mat,vital_mat,inter_mask.type(torch.int64),vital_mask.type(torch.int64)),(inter_mat_neg,vital_mat_neg,inter_neg_mask.type(torch.int64),vital_neg_mask.type(torch.int64)),torch.squeeze(torch.LongTensor([y]))

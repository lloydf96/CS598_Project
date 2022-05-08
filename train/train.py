import sys
import os
DIR_PATH = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(DIR_PATH,'data')
RESULTS_PATH = os.path.join(DIR_PATH,'results','results')
sys.path.append(DATA_PATH)
sys.path.append(os.path.join(DIR_PATH,'train'))

from pandas import read_pickle
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from loss import *
from BiLSTM import *
from Transformers import  *
from dataset import *
torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore")

def fit_and_evaluate(net, optimizer, loss_func, train_dl, test_dl, n_epochs, num_workers = 1 ):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    loss_func = loss_func.to(device)
    results = {}
    for i in range(n_epochs):
        print("Began epoch ",i)
        
        with torch.no_grad():
#             '''
#             Model is saved here
#             '''
            start_time = time.time()         
            train_loss = 0
        
        net.train()
        optimizer.zero_grad() #comment this line for op
        
        for step,(pos,neg,y) in enumerate(train_dl):
            pos,neg,y = tuple([ele.to(device) for ele in pos]),tuple([ele.to(device) for ele in neg]),y.to(device)
        
            net_output= net(pos,neg)
            loss = loss_func(net_output, y)
            with torch.no_grad():
                if loss is None:
                    break
                    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm = 1,norm_type = 'inf')
            optimizer.step()
                
            with torch.no_grad():
                # print(step)
                # print(net.fc[0].weight)
                # print(net.fc[0].weight.grad)
                # print('\n')
                train_loss += loss
                if step > 20:
                    break        
                optimizer.zero_grad()
                     
        with torch.no_grad():
            print(train_loss) 
            print("for %d step, time_elapsed for epoch %d is %d" %(step,i, time.time() - start_time))
            
        #scheduler.step()
        net.eval()
        start_time = time.time()
        with torch.no_grad():

            dev_loss,dev_prauc,dev_rocauc =  epoch_loss(net, loss_func, test_dl,device)
            
            print("Epoch: %s, Stored training loss: %s"%(i,train_loss))
            print("Epoch: %s, Testing loss: %s " %(i,dev_loss))
            print("Epoch: %s, Testing ROCAUC is %s and PRAUC is %s" %(i,dev_rocauc,dev_prauc))
            print("\n")
            
            results[i] = {'prauc' : dev_prauc,'rocauc' : dev_rocauc}
            
        print("Evaluation time is %s"%(time.time() - start_time))
        
    return results
        
        
def epoch_loss(net,loss_func,dl,device):
    batch = 0
    loss = 0
    i = True
    
    for step,(pos,neg,y) in enumerate(dl):
        pos,neg,y = tuple([ele.to(device) for ele in pos]),tuple([ele.to(device) for ele in neg]),y.to(device)

        net_output= net(pos,neg)
        loss = loss_func(net_output, y)
        batch += pos[0].shape[0]
        
        if type(net_output) == tuple:
            predict_proba = net_output[0]
        else:
            predict_proba = net_output
            
        predict_proba = F.softmax(predict_proba,dim = 1)
        
        if i:
            predict_proba_ds = predict_proba[:,1]
            y_ds = y
            i = False
        else:
            predict_proba_ds = torch.cat([predict_proba_ds,predict_proba[:,1]])
            y_ds = torch.cat([y_ds,y])

    y_ds,predict_proba_ds = y_ds.to('cpu'),predict_proba_ds.to('cpu')
   # print(confusion_matrix(z_list.to('cpu'),z_op_list.to('cpu')))
    return loss/batch,prauc(y_ds,predict_proba_ds),rocauc(y_ds,predict_proba_ds)
             
def epoch_auc(net,dl,device):
    batch = 0
    i = True
    
    for step,(pos,neg,y) in enumerate(dl):
        pos,neg,y = tuple([ele.to(device) for ele in pos]),tuple([ele.to(device) for ele in neg]),y.to(device)
        net_output= net(pos,neg)
        batch += pos[0].shape[0]
        
        
        if type(net_output) == tuple:
            predict_proba = net_output[0]
        else:
            predict_proba = net_output
            
        predict_proba = F.softmax(predict_proba,dim = 1).detach()
        y = y.detach()
        if i:
            predict_proba_ds = predict_proba[:,1]
            y_ds = y
            i = False
        else:
            predict_proba_ds = torch.cat([predict_proba_ds,predict_proba[:,1]])
            y_ds = torch.cat([y_ds,y])

    y_ds,predict_proba_ds = y_ds.to('cpu'),predict_proba_ds.to('cpu')
   # print(confusion_matrix(z_list.to('cpu'),z_op_list.to('cpu')))
    return prauc(y_ds,predict_proba_ds),rocauc(y_ds,predict_proba_ds)
    
def prauc(y,y_pred):
    precision,recall,threshold = precision_recall_curve(y,y_pred)
    return auc(recall,precision)

def rocauc(y,y_pred):
    return roc_auc_score(y,y_pred)

def get_data(batch_size = 4,task = 'mort_icu',data_test = False,N_neg = 32):
    
    patients_train = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'patients_train')
    vitals_train = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'vitals_train')
    intervention_train = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'intervention_train')
    y_train = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'y_train')
    
    if data_test:
        patients_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'patients_test')
        vitals_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'vitals_test')
        intervention_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'intervention_test')
        y_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'y_test')
    else:
        patients_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'patients_dev')
        vitals_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'vitals_dev')
        intervention_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'intervention_dev')
        y_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'y_dev')
        
    vitals_train = vitals_train.loc[:,(slice(None),['mean','mask'])]
    vitals_dev = vitals_dev.loc[:,(slice(None),['mean','mask'])]

#     subject_dev = patients_dev.index.get_level_values(0).drop_duplicates()
#     vitals_dev = vitals_dev.loc[subject_dev]
#     intervention_dev = intervention_dev.loc[subject_dev]
#     y_dev = y_dev.loc[subject_dev]
    
    batch_size = batch_size
    train_ds = EHRDataset(intervention_train,vitals_train,patients_train,y_train,task = task,N_neg = N_neg)
    train_dl = DataLoader(dataset = train_ds, collate_fn = collate_fn, batch_size = batch_size)

    dev_ds = EHRDataset(intervention_dev,vitals_dev,patients_dev,y_dev,task = task,N_neg = N_neg)
    dev_dl = DataLoader(dataset = dev_ds, collate_fn = collate_fn, batch_size = batch_size*4)
    
    return train_ds,train_dl,dev_ds,dev_dl

def train(train_ds,train_dl,dev_ds,dev_dl,model,loss_func,lr,weight_decay,n_epochs = 30):

    net = nn.DataParallel(model(train_ds.vital_signs.shape[1],train_ds.intervention.shape[1]))
    
    optimizer = optim.Adam(net.parameters(),lr = lr,weight_decay = weight_decay)
    training_results = fit_and_evaluate(net, optimizer,loss_func, train_dl, dev_dl, n_epochs, num_workers = 1 )
    
    training_results = pd.DataFrame.from_dict(training_results).T
    get_max_result = training_results[training_results.prauc.max() == training_results.prauc]
    return {'lr': lr, 'weight_decay' : weight_decay, 'prauc' : get_max_result.prauc.item(), 'rocauc' : get_max_result.rocauc.item(), 'epoch' : get_max_result.index.item()}

def fit_and_evaluate_model(net, optimizer, loss_func, train_dl, test_dl, n_epochs, num_workers = 1 ):
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    loss_func = loss_func.to(device)
    results = {}
    for i in range(n_epochs):
        print("Began epoch ",i)
        
        with torch.no_grad():
#             '''
#             Model is saved here
#             '''
            start_time = time.time()         
            train_loss = 0
        
        net.train()
        optimizer.zero_grad() #comment this line for op
        
        for step,(pos,neg,y) in enumerate(train_dl):
            pos,neg,y = tuple([ele.to(device) for ele in pos]),tuple([ele.to(device) for ele in neg]),y.to(device)
        
            net_output= net(pos,neg)
            loss = loss_func(net_output, y)
            with torch.no_grad():
                if loss is None:
                    break
                    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm = 1,norm_type = 'inf')
            optimizer.step()
                
            with torch.no_grad():
                # print(step)
                # print(net.fc[0].weight)
                # print(net.fc[0].weight.grad)
                # print('\n')
                train_loss += loss
                if step > 20:
                    break        
                optimizer.zero_grad()
                     
        with torch.no_grad():
            print(train_loss) 
            print("for %d step, time_elapsed for epoch %d is %d" %(step,i, time.time() - start_time))
            
        #scheduler.step()
        net.eval()
        start_time = time.time()
        with torch.no_grad():

            dev_loss,dev_prauc,dev_rocauc =  epoch_loss(net, loss_func, test_dl,device)
            
            print("Epoch: %s, Stored training loss: %s"%(i,train_loss))
            print("Epoch: %s, Testing loss: %s " %(i,dev_loss))
            print("Epoch: %s, Testing ROCAUC is %s and PRAUC is %s" %(i,dev_rocauc,dev_prauc))
            print("\n")
            
            results[i] = {'prauc' : dev_prauc,'rocauc' : dev_rocauc}
            
        print("Evaluation time is %s"%(time.time() - start_time))
        
    return results,net

def train_model(train_ds,train_dl,dev_ds,dev_dl,model,loss_func,lr,weight_decay,n_epochs = 30):

    net = nn.DataParallel(model(train_ds.vital_signs.shape[1],train_ds.intervention.shape[1]))
    
    optimizer = optim.Adam(net.parameters(),lr = lr,weight_decay = weight_decay)
    results,model = fit_and_evaluate_model(net, optimizer,loss_func, train_dl, dev_dl, n_epochs, num_workers = 1 )
    training_results = pd.DataFrame.from_dict(results).T
    get_max_result = training_results[training_results.prauc.max() == training_results.prauc]

    return {'lr': lr, 'weight_decay' : weight_decay, 'prauc' : get_max_result.prauc.item(), 'rocauc' : get_max_result.rocauc.item(), 'epoch' : get_max_result.index.item()},model


def get_data_partial(batch_size = 4,task = 'mort_icu',data_test = False,N_neg = 32,omit = 'vitals'):
    
    if data_test:
        patients_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'patients_test')
        vitals_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'vitals_test')
        intervention_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'intervention_test')
        y_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'y_test')
    else:
        patients_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'patients_dev')
        vitals_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'vitals_dev')
        intervention_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'intervention_dev')
        y_dev = pd.read_hdf(os.path.join(DATA_PATH,'mimic3Processed.h5'),'y_dev')
        
    vitals_dev = vitals_dev.loc[:,(slice(None),['mean','mask'])]

#     subject_dev = patients_dev.index.get_level_values(0).drop_duplicates()
#     vitals_dev = vitals_dev.loc[subject_dev]
#     intervention_dev = intervention_dev.loc[subject_dev]
#     y_dev = y_dev.loc[subject_dev]
    if omit == 'vitals':
        vitals_dev.loc[:,(slice(None),['mask'])] = 1
        vitals_dev.loc[:,(slice(None),['mean'])] = 0
    else: 
        intervention_dev = pd.DataFrame(data = np.zeros(intervention_dev.shape),columns = intervention_dev.columns,index = intervention_dev.index)
        
    batch_size = batch_size
    dev_ds = EHRDataset(intervention_dev,vitals_dev,patients_dev,y_dev,task = task,N_neg = N_neg)
    dev_dl = DataLoader(dataset = dev_ds, collate_fn = collate_fn, batch_size = batch_size)
    
    return dev_ds,dev_dl

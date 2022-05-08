import sys
import os
DIR_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_PATH = os.path.join(DIR_PATH,'data')
MODEL_PATH = os.path.join(DIR_PATH,'results','model')
RESULTS_PATH = os.path.join(DIR_PATH,'results','results')
sys.path.append(DATA_PATH)
sys.path.append(os.path.join(DIR_PATH,'train'))

import pandas as pd
from train import *
from BiLSTM import *
from Transformers import *
from loss import *
import json
import warnings
import torch
warnings.filterwarnings("ignore")

def test_model(model,model_name,hParams,batch_size,train_ds,train_dl,test_ds,test_dl,loss_func):
    lr = hParams.loc[model_name,'lr']
    weight_decay = hParams.loc[model_name,'weight_decay']
    n_epochs = 25
    
    results,model = train_model(train_ds,train_dl,test_ds,test_dl,model = model,loss_func = loss_func,lr = lr,weight_decay = weight_decay,n_epochs = n_epochs)
    
    torch.save(model.state_dict(), os.path.join(MODEL_PATH,model_name+'.pth'))
    return results
    
    
if __name__ == "__main__":
    
    hParams = pd.read_pickle(os.path.join(RESULTS_PATH,'hyperparameter_tuning_mi.pkl'))
    results = {}
    batch_size = 16
    train_ds,train_dl,test_ds,test_dl =  get_data(batch_size = batch_size,task = 'mort_icu',data_test = True)
    
    model = LF_pool_BiLSTM
    model_name = 'LF_pool_BiLSTM_mi'
    loss_func = nn.CrossEntropyLoss(reduction = 'mean')
    results_model = test_model(model,model_name,hParams,batch_size,train_ds,train_dl,test_ds,test_dl,loss_func)
    results[model_name] = results_model
    
    model = EFE_Transformer
    model_name = 'EFE_Transformer_mi'
    loss_func = nn.CrossEntropyLoss(reduction = 'mean')
    results_model = test_model(model,model_name,hParams,batch_size,train_ds,train_dl,test_ds,test_dl,loss_func)
    results[model_name] = results_model
    
    model = EFI_Transformer
    model_name = 'EFI_Transformer_mi'
    loss_func = nn.CrossEntropyLoss(reduction = 'mean')
    results_model = test_model(model,model_name,hParams,batch_size,train_ds,train_dl,test_ds,test_dl,loss_func)
    results[model_name] = results_model

    model = EFE_BiLSTM
    model_name = 'EFE_BiLSTM_mi'
    loss_func = nn.CrossEntropyLoss(reduction = 'mean')
    results_model = test_model(model,model_name,hParams,batch_size,train_ds,train_dl,test_ds,test_dl,loss_func)
    results[model_name] = results_model
    
    model = EFI_BiLSTM
    model_name = 'EFI_BiLSTM_mi'
    loss_func = nn.CrossEntropyLoss(reduction = 'mean')
    results_model = test_model(model,model_name,hParams,batch_size,train_ds,train_dl,test_ds,test_dl,loss_func)
    results[model_name] = results_model
    
    model = LF_BiLSTM
    model_name = 'LF_BiLSTM_mi'
    loss_func = nn.CrossEntropyLoss(reduction = 'mean')
    results_model = test_model(model,model_name,hParams,batch_size,train_ds,train_dl,test_ds,test_dl,loss_func)
    results[model_name] = results_model
    

    with open(os.path.join(RESULTS_PATH,'results_test_set_baseline.json'), 'w') as fp:
        json.dump(results, fp)

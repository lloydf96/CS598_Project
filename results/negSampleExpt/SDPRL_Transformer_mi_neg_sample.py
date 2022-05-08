import sys
import os
DIR_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_PATH = os.path.join(DIR_PATH,'data')
RESULTS_PATH = os.path.join(DIR_PATH,'results','results')
sys.path.append(DATA_PATH)
sys.path.append(os.path.join(DIR_PATH,'train'))

import pandas as pd
from train import *
from loss import *
from Transformers import *
import json
import warnings
warnings.filterwarnings("ignore")
  
if __name__ == "__main__":
    
    n_epochs = 20
    batch_size = 16
    
    loss_func =SDPRL_Loss()
    model = SDPRL_Transformer
    results = {}
    results_model = {}
    j = 0
    model_name = 'SDPRL_Transformer_mi'
    hParams = pd.read_pickle(os.path.join(RESULTS_PATH,'hyperparameter_tuning_mi.pkl'))
    lr = hParams.loc[model_name,'lr']*0.1
    weight_decay = hParams.loc[model_name,'weight_decay']
    
    
    for N_neg in [8,16,32,64,128]:
        train_ds,train_dl,dev_ds,dev_dl =  get_data(batch_size = batch_size,task = 'mort_icu',N_neg = 32)
        results_model[N_neg] = train(train_ds,train_dl,dev_ds,dev_dl,model = model,loss_func = loss_func,lr = lr,weight_decay = weight_decay,n_epochs = n_epochs)
        j = j + 1;
    
        with open(os.path.join(RESULTS_PATH,'results_SDPRL_Transformer_neg_batch.json'), 'w') as fp:
            json.dump(results_model, fp)
            

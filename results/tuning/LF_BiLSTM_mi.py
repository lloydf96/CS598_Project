import sys
import os
DIR_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_PATH = os.path.join(DIR_PATH,'data')
RESULTS_PATH = os.path.join(DIR_PATH,'results','results')
sys.path.append(DATA_PATH)
sys.path.append(os.path.join(DIR_PATH,'train'))

import pandas as pd
from train import *
from BiLSTM import *
import json
import warnings
warnings.filterwarnings("ignore")


    
if __name__ == "__main__":
    
    n_epochs = 35
    batch_size = 16
    train_ds,train_dl,dev_ds,dev_dl =  get_data(batch_size = batch_size,task = 'mort_icu')
    loss_func =nn.CrossEntropyLoss(reduction = 'mean')
    model = LF_BiLSTM
    results = {}
    results_model = {}
    j = 0
    
    for lr in [0.00001,0.00005,0.0001]:
        for weight_decay in [0.0005,0.001,0.005]:
            results_model[j] = train(train_ds,train_dl,dev_ds,dev_dl,model = model,loss_func = loss_func,lr = lr,weight_decay = weight_decay,n_epochs = n_epochs)
            j = j + 1;
     
    results['LF_BiLSTM'] = results_model
    
    with open(os.path.join(RESULTS_PATH,'results_LF_BiLSTM_mi.json'), 'w') as fp:
        json.dump(results_model, fp)
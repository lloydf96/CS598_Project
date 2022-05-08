import sys
import os
DIR_PATH = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_PATH = os.path.join(DIR_PATH,'data')
RESULTS_PATH = os.path.join(DIR_PATH,'results','results')
MODEL_PATH = os.path.join(DIR_PATH,'results','model')
sys.path.append(DATA_PATH)
sys.path.append(os.path.join(DIR_PATH,'train'))

import pandas as pd
from train import *
from Transformers import *
from loss import *
import json
import warnings
warnings.filterwarnings("ignore")


    
if __name__ == "__main__":
    
    n_epochs = 35
    batch_size = 16
    results = {}
    for omit in ['vitals','inter']:
    
        test_ds,test_dl =  get_data_partial(batch_size = batch_size,task = 'mort_icu',omit = omit)

        model = LF_pool_BiLSTM
        model = model(test_ds.vital_signs.shape[1],test_ds.intervention.shape[1]).to('cuda')
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'LF_pool_BiLSTM_mi.pth')))
        model.eval()
        prauc,rocauc = epoch_auc(model,test_dl,'cuda')
        results[omit+'_missing_LF_pool_BiLSTM'] = {'prauc': prauc,'rocauc': rocauc}

        model = LF_pool_Transformer
        model = model(test_ds.vital_signs.shape[1],test_ds.intervention.shape[1]).to('cuda')
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'LF_pool_Transformer_mi.pth')))
        model.eval()
        prauc,rocauc = epoch_auc(model,test_dl,'cuda')
        results[omit+'_missing_LF_pool_Transformer'] = {'prauc': prauc,'rocauc': rocauc}

        model = SDPRL_BiLSTM
        model = model(test_ds.vital_signs.shape[1],test_ds.intervention.shape[1]).to('cuda')
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'SDPRL_BiLSTM_mi.pth')))
        model.eval()
        prauc,rocauc = epoch_auc(model,test_dl,'cuda')
        results[omit+'_missing_SDPRL_BiLSTM'] = {'prauc': prauc,'rocauc': rocauc}

        model = SDPRL_Transformer
        model = model(test_ds.vital_signs.shape[1],test_ds.intervention.shape[1]).to('cuda')
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'SDPRL_Transformer_mi.pth')))
        model.eval()
        prauc,rocauc = epoch_auc(model,test_dl,'cuda')
        results[omit+'_missing_SDPRL_Transformer'] = {'prauc': prauc,'rocauc': rocauc}
    
    with open(os.path.join(RESULTS_PATH,'results_missing_expt.json'), 'w') as fp:
        json.dump(results, fp)

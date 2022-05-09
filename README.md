# Learning Robust Patient Representations from Multi-modal Electronic Health Records: A Supervised Deep Learning Approach

This repository is an attempt at reproducing the paper [Learning Robust Patient Representations from Multi-modal Electronic Health Records: A Supervised Deep Learning Approach](https://epubs.siam.org/doi/10.1137/1.9781611976700.66). 

The entire training and inference workflow is organised in different folders as follows:
    
    ├── data
    │   └── data_preprocessing.ipynb    # Data Preprocessing
    ├── results                         
    │   ├── missingModal                # Contains .py files to test for missing modal peformance
    │   ├── model                       # folder contains trained models
    │   ├── results                     # Contains results
    │   │    └── tuning_results.ipynb   # Compiles tuning results
    │   ├── test                        # Trains models based on hyperparameter tuning results and tests on testing set
    │   ├── swb_files                   # .swb files used for training models in batches
    │   │    └── output                 # Contains output files from batch jobs
    │   └── tuning                      # Hyperparameter tuning on learning rate and weight decay for optimizer
    └── train                           # Contains PyTorch Dataset and model objects

All the models were run on [HAL](https://wiki.ncsa.illinois.edu/display/ISL20/HAL+cluster) using batch files in swb_files

## Requirements

To create conda environment:

```setup
conda create --name <env> --file requirements.txt
```

## Tuning
To find weight decay and learning rate hyperparameters for Adam optimizer run all the .py files in folder results -> tuning:

```tuning
cd results/tuning
python <model_name>_mi.py 
```

## Train the model on hyperparameters and test on testing set
To train the model on hyperparameters and test on testing set first go to results -> results and run the tuning_results.ipynb
Then run all the .py files in results>test

```test
cd results/test
python test_baseline.py
python test_SDPRL.py
```

All the models will be saved in results -> model folder


## Check the effect of missing modal in SDPRL

Run the .py files in results -> missingModal

```missing_modal
cd results/missingModal
python missing_expt.py
```

## Results

## Model performance

### Model and baseline results Area Under Precision Recall Curve

|Model Encoder|Framework |Reproduced AUPRC| Model AUPRC|
|-------------|----------|-----|-----|
|BiLSTM       |EFE       |31.67|52.65|
|             |EFI       |30.62|52.54|
|             |LF        |31.72|51.67|
|             |LF_pool   |48.08|51.82|
|             |SDPRL     |48.96|52.77|
|Transformer  |EFE       |32.55|52.31|
|             |EFI       |28.3 |52.1 |
|             |LF        |32.57|52.61|
|             |LF_pool   |39.32|52.68|
|             |SDPRL     |47.35|53.66|


### Model and baseline results Area Under Receiver Operating curve


|Model Encoder|Framework |Reproduced AUCROC  | Model AUCROC |
|-------------|----------|--------|-------|
|BiLSTM       |EFE       |79      |89.89  |
|             |EFI       |78.01   |90.3   |
|             |LF        |78.09   |89.66  |
|             |LF_pool   |85.4    |89.49  |
|             |SDPRL     |82.04   |90.24  |
|Transformer  |EFE       |73.81   |89.53  |
|             |EFI       |78.81   |89.58  |
|             |LF        |68.17   |89.93  |
|             |LF_pool   |82.44   |89.81  |
|             |SDPRL     |83.48   |90.6   |


## Durability under missing information

### Area Under Precision Recall Curve
|Missing Mode|Model|Reproduced AUPRC|Paper AUPRC|
|-------------|----------|--------|-------|
|Vital Signs|LF-Pool + BiLSTM|39.75|28.57|
||SDPRL + BiLSTM|49.36|31.86|
||LF-Pool + Transformer|34.43|22.22|
||SDPRL + Transformer|40.99|30.13|
|Intervention|LF-Pool + BiLSTM|26.28|50.27|
||SDPRL + BiLSTM|24.03|50.58|
||LF-Pool + Transformer|16.09|51.1|
||SDPRL + Transformer|18.4|51.47|



### Area Under Receiver Operating curve







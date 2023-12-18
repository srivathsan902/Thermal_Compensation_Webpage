import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_training import model_training

def process_data(temp_df, displacement_df):
    temp_keys = list(temp_df.keys())
    temp_data_keys = {key: 'Day '+str(i+1) for i, key in enumerate(temp_keys)}
    temp_data = {temp_data_keys[key]: value for key, value in temp_df.items()}
    
    displacement_keys = list(displacement_df.keys())
    displacement_data_keys = {key: 'Day '+str(i+1) for i, key in enumerate(displacement_keys)}
    displacement_data = {displacement_data_keys[key]: value for key, value in displacement_df.items()}
    
    return temp_data, displacement_data

def training_data_v1(data, test_split):
    days_keys = list(data.keys())
    tot_days = len(days_keys)
    
    test_days = int(test_split*tot_days)
    train_days = tot_days - test_days
    
    train_df = pd.DataFrame()
    for sheet_name in days_keys[:train_days]:
        sheet_data = data[sheet_name]
        train_df = pd.concat([train_df, sheet_data], ignore_index = True)
    
    test_df = pd.DataFrame()

    for sheet_name in days_keys[train_days:]:
        sheet_data = data[sheet_name]
        test_df = pd.concat([test_df, sheet_data], ignore_index = True)
    
    return train_df, test_df


def output(model, file1, file2, error_threshold, test_split):
    # temp_names = ['T'+str(i+1) for i in range(6)]
    # temp_df = pd.read_excel(file1, names = temp_names, sheet_name = None)
    # displacement_df = pd.read_excel(file2, names = ['Dia Disp'], sheet_name = None)
    
    # temp_data, displacement_data = process_data(temp_df, displacement_df)
    
    # train_X, test_X = training_data_v1(temp_data, 0.01*test_split)
    # train_Y, test_Y = training_data_v1(displacement_data, 0.01*test_split)
    
    # training_data_X = [train_X, test_X]
    # training_data_Y = [train_Y, test_Y]
    
    # results = model_training(model, training_data_X, training_data_Y, error_threshold, temp_data, displacement_data, 0.01*test_split)
    results=10
    return str(results)

# model_options = ["Linear Regression", "Random Forest", "Support Vector Machine", "Neural Networks"]
# output('Neural Networks','Datasets\ACE_X.xlsx','Datasets\ACE_Y.xlsx',20)
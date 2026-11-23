import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

'''
Site I used for reference: https://www.geeksforgeeks.org/data-analysis/data-preprocessing-machine-learning-python/
'''

'''
Data: not model-ready yet. Model_Dataset.csv is the right input. The models need it after these steps:

Drop the VIX Volume column.
Log transforms.
Split into train, 2024 validation and 2025-26 test.
Clip with train-set bounds.
StandardScaler fit on train only.
'''

def split_data(df, test_year=2025):
    df = df.sort_index()  # Ensure the DataFrame is sorted by date
    
    train_df = df[ df.index.year < test_year ] 
    test_df = df[ df.index.year >= test_year ] 

    # Ranges (report it in the write-up)
    print(f"Train: {train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df)} rows)")
    print(f"Test:  {test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df)} rows)")

    return train_df, test_df


def preprocessing(df): # df is the dataset created in featureCreation.py
            #       Data Preprocessing    #
            #       Feature Scaling       #
            #       Outlier Removal       #

    '''
    The idea is:
    1. Check for missing values and data types of each column
    2. Remove outliers using the Interquartile Range (IQR) method
    '''
    print(df.info()) # count of non-null entries and data type of each column.
    print("--------------------------------")
    print(df.isnull().sum()) # Returns the number of missing values per column.
    print("--------------------------------")
    # df.describe(): Computes count, mean, std deviation, min/max and quartiles for numerical columns.
    print(df.describe())

    '''
    Remove Outliers Using the Interquartile Range (IQR) Method
    IQR = Q3 (75th percentile) - Q1 (25th percentile).

    Values below Q1 - 1.5IQR or above Q3 + 1.5IQR are outliers.
    '''

    # features in README as well
    features = [
        '1D_Return', '5D_Return', '20D_Return', 
        '20D_Volatility', 'Dist_50D_MA', 
        'Up_Day_Streak', 'Overnight_Gap', 
        'RSI', 'Rel_Volume', 'QQQ_1D_Return', 
        'VIX_Level', 'VIX_1D_Change', 
        'SMH_1D_Return'
        ]

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        # IQR = Q3 (75th percentile) - Q1 (25th percentile)
        iqr = q3 - q1

        low = q1 - 1.5 * iqr # Lower bound for outliers
        high = q3 + 1.5 * iqr # Upper bound for outliers
        
        # returns boolean on each row, if false, create new dataframe without the outliers by removing them
        df = df[ df[feature].between(low, high) ] 

    '''
    Separate Target Variable from features & Feature scaling

        - scaling will be done using Min-Max scaling 


    '''

    y = df['Target'] #--> Grab target as seperate column
    x = df.drop(columns=['Target']) # remove target from features

    return df

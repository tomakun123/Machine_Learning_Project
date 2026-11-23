import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

'''
Site I used for reference: https://www.geeksforgeeks.org/data-analysis/data-preprocessing-machine-learning-python/
'''

def split_data(df, validate_year = 2024, test_year=2025):
    df = df.sort_index()  # Ensure the DataFrame is sorted by date
    
    train_df = df[ df.index.year < validate_year ] 
    validate_df = df[ df.index.year == validate_year]
    test_df = df[ df.index.year >= test_year ] 

    # Ranges (report it in the write-up)
    print(f"Train: {train_df.index.min().date()} to {train_df.index.max().date()} ({len(train_df)} rows)")
    print(f"Validate: {validate_df.index.min().date()} to {validate_df.index.max().date()} ({len(validate_df)} rows)")
    print(f"Test:  {test_df.index.min().date()} to {test_df.index.max().date()} ({len(test_df)} rows)")

    return train_df, validate_df, test_df


def preprocessing(dfo): # df is the dataset created in featureCreation.py
    '''
    The idea is:
    1. Check for missing values and data types of each column
    2. Log transformations (log(x): Rel_Volume, VIX_Level, 20D_Volatility; log1p(x): VIX_1D_Change (can be negative)) --> found from feature distributions
    3. Split data into train (2024 > year), validation (2024), testing (2025 <= year)
    4. clip with train bounds (cap on extreme values so it doesn't blow up data) --> found from feature distributions
    5. Standardize data into mean=0, SD=1 to ensure features stand equally, and one doesn't blow up the data
    '''
    print(dfo.info()) # count of non-null entries and data type of each column.
    print("--------------------------------")
    print(dfo.isnull().sum()) # Returns the number of missing values per column.
    print("--------------------------------")
    # df.describe(): Computes count, mean, std deviation, min/max and quartiles for numerical columns.
    print(dfo.describe())

    # features in README as well
    features = [
        '1D_Return', '5D_Return', '20D_Return', 
        '20D_Volatility', 'Dist_50D_MA', 
        'Up_Day_Streak', 'Overnight_Gap', 
        'RSI', 'Rel_Volume', 'QQQ_1D_Return', 
        'VIX_Level', 'VIX_1D_Change', 
        'SMH_1D_Return'
        ]

    '''
    Log transformations on Rel_Volume, VIX_Level, 20D_Volatility, VIX_1D_Change
    '''
    df = dfo.copy() # copy so we don't modify original data
    for c in ['Rel_Volume', 'VIX_Level', '20D_Volatility']:
        df[c] = np.log(df[c])
    df['VIX_1D_Change'] = np.log1p(df['VIX_1D_Change'])

    ### Splitting data into train, validate, test ###
    train_df, validate_df, test_df = split_data(df, 2024, 2025)
    # seperating features from target since we dont want to clip or standardize the target
    X_train,    y_train    = train_df[features],    train_df['Target']
    X_validate, y_validate = validate_df[features], validate_df['Target']
    X_test,     y_test     = test_df[features],     test_df['Target']

    #################################################

    '''
    clipping training bounds
    --> Clip on training data only, apply rules to train, validation, and test to avoid data leakage
    '''
    lo = X_train.quantile(0.01) # 1st percentile
    hi = X_train.quantile(0.99) # 99th percentile
    
    lo['Up_Day_Streak'], hi['Up_Day_Streak'] = -5, 5 # +-5 streak rule to not blow up data
    
    X_train = X_train.clip(lo, hi, axis=1) # axis=1 --> columns, since we apply clippings to features, not rows
    X_validate = X_validate.clip(lo, hi, axis=1)
    X_test = X_test.clip(lo, hi, axis=1)

    '''
    Standardization of data --> mean = 0, standard deviation = 1 for all features
     - allows each feature to be in similar ranges, 
    '''
    scaler = StandardScaler() # empty scalar, using fit_transform to populate it
    X_train    = scaler.fit_transform(X_train)   # fit_transform --> compute each feature's mean and standard deviation from given data and store it
    X_validate = scaler.transform(X_validate)    # transform --> apply stored mean and std to convert data with z = (x − mean) / std 
    X_test     = scaler.transform(X_test)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

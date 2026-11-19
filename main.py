from yahooFinance import download_data
from featureCreation import feature_creation
from data_preprocessing import preprocessing, split_data

def main():
    features = [
            '1D_Return', '5D_Return', '20D_Return', 
            '20D_Volatility', 'Dist_50D_MA', 
            'Up_Day_Streak', 'Overnight_Gap', 
            'RSI', 'Rel_Volume', 'QQQ_1D_Return', 
            'VIX_Level', 'VIX_1D_Change', 
            'SMH_1D_Return'
            ]

    download_data() # retrieves data from Yahoo Finance and saves as CSV files in /Datasets

    df = feature_creation()
    print("df", df.head())

    '''
    Preprocess data + Split data into train/validate & test sets by year (2025 is default)
    '''
    pp_df = preprocessing(df)
    train_df, test_df = split_data(pp_df, 2025)
    print("pre_processed_data: ", pp_df.head(), 
          "train: ", train_df.head(), 
          "test: ", test_df.head())

    x_train, y_train = train_df[features], train_df['Target']
    x_test,  y_test  = test_df[features],  test_df['Target']

    '''
    Now scaling will be added later
    '''
    # results = train_models(*splits)        # step 3: add later for Section D

if __name__ == "__main__":
    main()
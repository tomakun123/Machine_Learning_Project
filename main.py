from yahooFinance import download_data
from featureCreation import feature_creation
from data_preprocessing import preprocessing

def main():

    download_data() # retrieves data from Yahoo Finance and saves as CSV files in /Datasets

    df = feature_creation() # Create features based on CSV files downlaoded in /Datasets and output Model_Dataset.csv
    print("df", df.head())

    '''
    Preprocess data (data_preprocess.py) --> returns split data with following changes:
        1. Check for missing values and data types of each column
        2. Log transformations (log(x): Rel_Volume, VIX_Level, 20D_Volatility; log1p(x): VIX_1D_Change (can be negative)) --> found from feature distributions
        3. Split data into train (2024 > year), validation (2024), testing (2025 <= year)
        4. clip with train bounds (cap on extreme values so it doesn't blow up data) --> found from feature distributions
        5. Standardize data into mean=0, SD=1 to ensure features stand equally, and one doesn't blow up the data
    '''
    X_train, X_validate, X_test, y_train, y_validate, y_test = preprocessing(df)

    # Section D: train models


if __name__ == "__main__":
    main()
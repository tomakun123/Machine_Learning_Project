from yahooFinance import download_data
from featureCreation import feature_creation
from data_preprocessing import preprocessing

def main():
    download_data() # retrieves data from Yahoo Finance and saves as CSV files in /Datasets
    df = feature_creation()
    print("df", df.head())
    splits = preprocessing(df)
    print("splits", splits)
    # results = train_models(*splits)        # step 3: add later for Section D

if __name__ == "__main__":
    main()
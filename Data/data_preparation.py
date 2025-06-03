import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    filename="\CryptoLab\logs/time_series_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class EDA:
    def __init__(self, path, coins):
        self.path = path
        self.coins = coins
        self.run()

    def run(self):
        """Main function to execute all steps."""
        for coin in self.coins:
            logging.info(
                "********************* EDA Results for {coin} ************************************"
            )
            print(
                "\033[93m⚠ Please note that most of the statistical information is provided in the log file! ⚠\033[0m"
            )

            df = self.load_data(coin)
            if df is None:
                continue

            df = self.clean_data(df)
            self.visualize_data(df, coin)

            self.timeseries_visual_analysis(df, title="Before Conversion")
            cleaned_df = self.timeseries_analysis(df)
            self.timeseries_visual_analysis(cleaned_df, title="After Conversion")

            self.save_data(coin, cleaned_df, folder="cleaned")

    def load_data(self, coin):
        """Load and preprocess data"""
        path = os.path.join(self.path, "data", "raw", f"{coin}.csv")
        try:
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            if "Adj Close" in df.columns:
                df.drop(columns=["Adj Close"], inplace=True)

            logging.info(f"Loaded data for {coin}")
            return df
        except Exception as e:
            logging.error(f"Error loading {coin} data: {e}")
            return None

    def clean_data(self, df):
        """Clean and preprocess data"""
        try:
            # Remove duplicates
            df = df[~df.index.duplicated(keep="first")]

            # Resample only if data is not already daily
            # if (df.index.freq is None) or (df.index.freq != 'D'):
            # df = df.resample('D').mean()

            df = df.apply(pd.to_numeric, errors="coerce")

            # Outlier Detection & Removal
            df = self.detect_outliers(df)

            # Handle missing values
            df.fillna(method="ffill", inplace=True)  # Forward fill
            df.fillna(method="bfill", inplace=True)  # Backward fill
            df.fillna(0, inplace=True)

            logging.info("Data cleaned successfully")
            return df
        except Exception as e:
            logging.error(f"Data cleaning error: {e}")
            
if __name__ == "__main__":
    repo_path = r"C:\CRNProject"
    coin_symbols = ["ETH", "XRP", "BNB", "LTC", "USDT"]
    EDA(project_path=repo_path, coin_list=coin_symbols)   
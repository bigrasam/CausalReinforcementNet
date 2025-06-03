import yfinance as yf
import os


class CRNDataIngestion:

    def __init__(self, tickers, project_root, frequency="1d", period="5y"):
        self.tickers = tickers
        self.frequency = frequency
        self.period = period
        self.project_root = project_root
        self.download_and_store()

    def download_and_store(self):
        for ticker in self.tickers:
            df = yf.download(ticker, period=self.period, interval=self.frequency)
            self.save_csv(ticker, df)

    def save_csv(self, ticker, df):
        output_dir = os.path.join(self.project_root, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(file_path)
        print(f"Saved {ticker} to {file_path}")



if __name__ == "__main__":
    # Update this path based on your local repository
    project_path = r"C:\CRNProject"
    
    # Combined list of cryptocurrencies and macro-financial assets
    tickers = [
        "ETH-USD",       # Ethereum
        "XRP-USD",       # Ripple
        "BNB-USD",       # Binance Coin
        "LTC-USD",       # Litecoin
        "USDT-USD",      # Tether
        "GC=F",          # Gold Futures
        "^GSPC",         # S&P 500
        "DX-Y.NYB",      # U.S. Dollar Index
        "URTH",          # MSCI World
        "CL=F"           # Crude Oil WTI
    ]

    CRNDataIngestion(tickers=tickers, project_root=project_path)


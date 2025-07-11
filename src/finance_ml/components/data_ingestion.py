import yfinance as yf
import pandas as pd

class DataIngestion:
    def __init__(self, config, ticker):
        self.config = config
        self.ticker = ticker

    def download_data(self):
        df = yf.download(
            tickers=self.ticker,
            period=self.config.period,
            interval=self.config.interval,
            auto_adjust=True
        )
        if df.empty:
            print(f"No data for {self.ticker}")
            return df
        
        df = df.reset_index()


        print("Columns after reset_index:", df.columns.tolist())
        print("First 5 rows:\n", df.head())


        df['Ticker'] = self.ticker
        df.columns = df.columns.droplevel(level=1)

        # Rename 'Date' to 'Datetime' to match your expected column names
        df = df.rename(columns={'Date': 'Datetime'})

        print("Columns after renaming:", df.columns.tolist())

        df = df[['Ticker', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

        df.to_csv(self.config.raw_data_file, index=False)
        print(f"Data saved to {self.config.raw_data_file}")
        return df
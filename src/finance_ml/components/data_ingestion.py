import yfinance as yf

class DataIngestion:
    def __init__(self, config, ticker):
        self.config = config
        self.ticker = ticker

    def download_data(self):
        df = yf.download(
            tickers=self.ticker,
            period=self.config.period,
            interval=self.config.interval
        )
        if df.empty:
            print(f"No data for {self.ticker}")
            return df
        
        df = df.reset_index()
        df['Ticker'] = self.ticker
        df.columns = df.columns.droplevel(level=1)
        df = df[['Ticker', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

        df.to_csv(self.config.raw_data_file, index=False)
        print(f"Data saved to {self.config.raw_data_file}")
        return df
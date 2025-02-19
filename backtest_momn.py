import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
from datetime import datetime, timedelta

def fetch_nse_stock_list():
    url = 'https://raw.githubusercontent.com/prayan2702/Streamlit-momn/main/NSE_EQ_ALL.csv'
    df = pd.read_csv(url)
    df['Yahoo_Symbol'] = df['Symbol'] + '.NS'
    
    valid_symbols = []
    total_symbols = len(df['Yahoo_Symbol'])
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(df['Yahoo_Symbol'].tolist()):
        try:
            test_data = yf.download(symbol, period="1d", progress=False)
            if not test_data.empty:
                valid_symbols.append(symbol)
        except Exception:
            pass  # Ignore symbols that cause errors
        
        progress_bar.progress((i + 1) / total_symbols)
    
    return valid_symbols

class MomentumStrategy(bt.Strategy):
    params = (('lookback', 252), ('top_n', 30))
    
    def __init__(self):
        self.stocks = []

    def next(self):
        if len(self.data.datetime.dates) % 21 == 0:
            self.rebalance()

    def rebalance(self):
        all_data = {}
        for d in self.datas:
            close_prices = pd.Series(d.get(size=self.params.lookback), index=pd.to_datetime(self.data.datetime.get(size=self.params.lookback)))
            all_data[d._name] = close_prices
        df = pd.DataFrame(all_data).dropna()
        
        momentum_scores = ((df.iloc[-1] / df.iloc[0]) - 1) * 100
        ranked_stocks = momentum_scores.sort_values(ascending=False).head(self.params.top_n).index.tolist()

        for d in self.datas:
            if d._name in ranked_stocks:
                self.order_target_percent(d, target=1.0 / len(ranked_stocks))
            else:
                self.order_target_percent(d, target=0.0)

def main():
    st.title("Momentum Strategy Backtesting")
    
    start_date = st.date_input("Start Date", datetime(2015, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 1, 1))
    
    if st.button("Run Backtest"):
        st.write("Fetching valid stock symbols... This may take a few minutes.")
        symbols = fetch_nse_stock_list()
        st.write(f"Total valid stocks found: {len(symbols)}")
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(MomentumStrategy)
        
        progress_bar = st.progress(0)
        valid_data = []
        for i, symbol in enumerate(symbols):
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    data_feed = bt.feeds.PandasData(dataname=data)
                    cerebro.adddata(data_feed)
                    valid_data.append(symbol)
            except Exception:
                pass  # Ignore errors while loading data
            progress_bar.progress((i + 1) / len(symbols))
        
        if not valid_data:
            st.error("No valid stock data available for backtesting.")
            return
        
        cerebro.broker.set_cash(100000)
        cerebro.run()
        st.write("Final Portfolio Value:", cerebro.broker.getvalue())
        cerebro.plot(style='candlestick')

if __name__ == "__main__":
    main()

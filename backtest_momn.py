import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
from datetime import datetime, timedelta

# Function to fetch NSE stock list from CSV
def fetch_nse_stock_list():
    url = 'https://raw.githubusercontent.com/prayan2702/Streamlit-momn/main/NSE_EQ_ALL.csv'
    df = pd.read_csv(url)
    df['Yahoo_Symbol'] = df['Symbol'] + '.NS'
    return df['Yahoo_Symbol'].tolist()

# Backtrader Strategy Class
class MomentumStrategy(bt.Strategy):
    params = (('lookback', 252), ('top_n', 30))
    
    def __init__(self):
        self.stocks = []

    def next(self):
        if len(self.data.datetime.dates) % 21 == 0:  # Monthly rebalance
            self.rebalance()

    def rebalance(self):
        all_data = {}
        for d in self.datas:
            close_prices = pd.Series(d.get(size=self.params.lookback), index=pd.to_datetime(self.data.datetime.get(size=self.params.lookback)))
            all_data[d._name] = close_prices
        df = pd.DataFrame(all_data).dropna()
        
        # Calculate Momentum (12M ROC)
        momentum_scores = ((df.iloc[-1] / df.iloc[0]) - 1) * 100
        ranked_stocks = momentum_scores.sort_values(ascending=False).head(self.params.top_n).index.tolist()

        # Rebalance Portfolio
        for d in self.datas:
            if d._name in ranked_stocks:
                self.order_target_percent(d, target=1.0 / len(ranked_stocks))
            else:
                self.order_target_percent(d, target=0.0)

# Streamlit UI
def main():
    st.title("Momentum Strategy Backtesting")
    
    # Input Parameters
    start_date = st.date_input("Start Date", datetime(2015, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 1, 1))
    
    if st.button("Run Backtest"):
        symbols = fetch_nse_stock_list()
        cerebro = bt.Cerebro()
        cerebro.addstrategy(MomentumStrategy)
        
        for symbol in symbols:
            data = bt.feeds.YahooFinanceData(dataname=symbol, fromdate=start_date, todate=end_date)
            cerebro.adddata(data)
        
        cerebro.broker.set_cash(100000)
        cerebro.run()
        st.write("Final Portfolio Value:", cerebro.broker.getvalue())
        cerebro.plot(style='candlestick')

if __name__ == "__main__":
    main()

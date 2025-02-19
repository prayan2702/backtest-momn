import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to fetch NSE stock list from CSV
def fetch_nse_stock_list(universe):
    if universe == 'N750':
        url = 'https://raw.githubusercontent.com/prayan2702/Streamlit-momn/main/ind_niftytotalmarket_list.csv'
    elif universe == 'AllNSE':
        url = 'https://raw.githubusercontent.com/prayan2702/Streamlit-momn/main/NSE_EQ_ALL.csv'
    else:
        url = f'https://raw.githubusercontent.com/prayan2702/Streamlit-momn/main/ind_{universe.lower()}list.csv'
    
    df = pd.read_csv(url)
    df['Yahoo_Symbol'] = df['Symbol'] + '.NS'
    return df['Yahoo_Symbol'].tolist()

# Momentum Ranking Logic
def getMedianVolume(data):
    return round(data.median(), 0)

def getDailyReturns(data):
    return data.pct_change(fill_method='ffill')

def getMaskDailyChange(data):
    m1 = getDailyReturns(data).eq(np.inf)
    m2 = getDailyReturns(data).eq(-np.inf)
    return getDailyReturns(data).mask(m1, data[~m1].max(), axis=1).mask(m2, data[~m2].min(), axis=1).bfill(axis=1)

def getStdev(data):
    return np.std(getMaskDailyChange(data) * 100)

def getAbsReturns(data):
    return (data.iloc[-1] / data.iloc[0] - 1) * 100

def getVolatility(data):
    return round(np.std(data) * np.sqrt(252) * 100, 2)

def getSharpeRoC(roc, volatility):
    return round(roc / volatility, 2)

def calculate_z_score(data):
    mean = data.mean()
    std = data.std()
    z_score = (data - mean) / std
    return z_score.round(2)

def rank_stocks(close_df, volume_df, high_df, end_date):
    dfStats = pd.DataFrame(index=close_df.columns)
    dfStats['Close'] = round(close_df.iloc[-1], 2)
    dfStats['dma200d'] = round(close_df.rolling(window=200).mean().iloc[-1], 2)
    dfStats['roc12M'] = getAbsReturns(close_df.iloc[-252:])
    dfStats['vol12M'] = getVolatility(getDailyReturns(close_df.iloc[-252:]))
    dfStats['sharpe12M'] = getSharpeRoC(dfStats['roc12M'], dfStats['vol12M'])
    dfStats['z_score12M'] = calculate_z_score(dfStats['sharpe12M'])
    dfStats['volm_cr'] = (getMedianVolume(volume_df.iloc[-252:]) / 1e7).round(2)
    dfStats['ATH'] = round(high_df.max(), 2)
    dfStats['AWAY_ATH'] = round((dfStats['Close'] / dfStats['ATH'] - 1) * 100, 2)
    dfStats['circuit'] = (getDailyReturns(close_df.iloc[-252:]) * 100).apply(lambda x: ((x == 4.99) | (x == 5.00) | (x == 9.99) | (x == 10.00) | (x == 19.99) | (x == 20.00) | (x == -4.99) | (x == -5.00) | (x == -9.99) | (x == -10.00) | (x == -19.99) | (x == -20.00)).sum())
    dfStats['circuit5'] = (getDailyReturns(close_df.iloc[-63:]) * 100).apply(lambda x: ((x == 4.99) | (x == 5.00) | (x == -4.99) | (x == -5.00)).sum())


    # Apply filters
    cond1 = dfStats['volm_cr'] > 1
    cond3 = dfStats['Close'] > dfStats['dma200d']
    cond4 = dfStats['roc12M'] > 6.5
    cond5 = dfStats['circuit'] < 20
    cond6 = dfStats['AWAY_ATH'] > -25
    cond7 = dfStats['roc12M'] < 1000
    cond8 = (dfStats['roc1M'] / dfStats['roc12M'] * 100) < 50
    cond9 = dfStats['Close'] > 30
    cond10 = dfStats['circuit5'] <= 10

    dfStats['final_momentum'] = cond1 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8 & cond9 & cond10
    filtered = dfStats[dfStats['final_momentum']].sort_values('z_score12M', ascending=False)
    filtered['Rank'] = range(1, len(filtered) + 1)

    return filtered

# Backtrader Strategy
class MomentumStrategy(bt.Strategy):
    def __init__(self):
        self.rank_threshold = 100
        self.portfolio_size = 30
        self.rebalance_dates = []

    def next(self):
        if self.data.datetime.date(0) in self.rebalance_dates:
            self.rebalance_portfolio()

    def rebalance_portfolio(self):
        # Get the current date
        current_date = self.data.datetime.date(0)

        # Calculate the start date for 12 months lookback
        start_date = current_date - timedelta(days=365)
        end_date = current_date

        # Get the list of symbols
        symbols = [d._name for d in self.datas]

        # Download historical data
        close_df, volume_df, high_df = get_ranking_data(symbols, start_date, end_date)

        # Rank the stocks
        ranked_stocks = rank_stocks(close_df, volume_df, high_df, end_date)

        # Select top 30 stocks
        top_stocks = ranked_stocks[ranked_stocks['Rank'] <= self.portfolio_size].index.tolist()

        # Rebalance the portfolio
        for data in self.datas:
            if data._name in top_stocks:
                self.order_target_percent(data, target=1.0 / self.portfolio_size)
            else:
                self.order_target_percent(data, target=0.0)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            pass

# Streamlit App
def main():
    st.title("Momentum Strategy Backtesting")

    # Select universe
    universe_options = ['Nifty50', 'Nifty100', 'Nifty200', 'Nifty250', 'Nifty500', 'N750', 'AllNSE']
    selected_universe = st.selectbox("Select Universe", universe_options, index=6)

    # Fetch NSE stock list
    symbols = fetch_nse_stock_list(selected_universe)

    # Date range for backtesting
    start_date = st.date_input("Start Date", datetime(2015, 1, 1))
    end_date = st.date_input("End Date", datetime(2023, 1, 1))

    # Run backtest
    if st.button("Run Backtest"):
        with st.spinner("Running Backtest..."):
            # Create a Cerebro engine instance
            cerebro = bt.Cerebro()

            # Add the strategy to Cerebro
            cerebro.addstrategy(MomentumStrategy)

            # Add data feeds for each stock in the universe
            for symbol in symbols:
                data = bt.feeds.YahooFinanceData(dataname=symbol, fromdate=start_date, todate=end_date)
                cerebro.adddata(data)

            # Set the initial cash
            cerebro.broker.set_cash(100000)

            # Set the commission
            cerebro.broker.setcommission(commission=0.001)

            # Run the backtest
            results = cerebro.run()

            # Display results
            st.success("Backtest completed!")
            st.write("Final Portfolio Value:", cerebro.broker.getvalue())

            # Plot the results
            st.write("Portfolio Performance:")
            cerebro.plot(style='candlestick')

if __name__ == "__main__":
    main()

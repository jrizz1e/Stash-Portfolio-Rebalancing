# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:38:29 2020

@author: Jonathan
"""

import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib
from matplotlib import pyplot as plt

tickers = pd.read_excel(r"C:\Users\Jonathan\Investing\StashETFs.xlsx")
tickers = tickers['Ticker'].to_list()

#get data for all tickers
closing_prices = pd.DataFrame()
failed = []
passed = []
for ticker in tickers:
    one_ticker = yf.Ticker(ticker)
    try:
        ticker_data = one_ticker.history(period="5y")
        print('sucessfully gathered data for ' + ticker)
        # combine all ticker data into df
        passed.append(ticker)
        closing_prices = pd.concat([closing_prices, ticker_data['Close']], axis=1)
    except:
        # keep track of tickers that failed
        failed.append(ticker)

# try to gather data for failed tickers until we have data for all of them
while len(failed) > 0:
    for ticker in failed:
        try:
            ticker_data = one_ticker.history(period="5y")
            print('sucessfully gathered data for ' + ticker)
            # combine all ticker data into df
            passed.append(ticker)
            closing_prices = pd.concat([closing_prices, ticker_data['Close']], axis=1)
            failed.remove(ticker)
        except:
            print(ticker + ' failed again')

closing_prices.columns = passed

closing_prices.to_csv(r"C:\Users\Jonathan\Investing\ETF_closing_prices.csv", na_rep=np.nan)

#read in price data
prices = pd.read_csv(r"C:\Users\Jonathan\Investing\ETF_closing_prices.csv")
dates = prices['Unnamed: 0']
#remove date to isolate prices
prices = prices.drop('Unnamed: 0', axis=1)
#calc daily returns ln(today/yesterday)
daily_return = np.log(prices/prices.shift(1))
#relabel columns
daily_return['Date'] = dates
cols = ['Date']+prices.columns.to_list()
daily_return = daily_return.reindex(columns=cols)
#write to csv
daily_return.to_csv(r"C:\Users\Jonathan\Investing\ETF_daily_returns.csv", na_rep=np.nan)

# Read in price data
df = pd.read_csv(r"C:\Users\Jonathan\Investing\ETF_closing_prices.csv", parse_dates=True, index_col="Unnamed: 0")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ef.save_weights_to_file(r"C:\Users\Jonathan\Investing\weights.csv")
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

weights = pd.read_csv(r"C:\Users\Jonathan\Investing\weights.csv", header=None)
weights.columns = ['Ticker', 'Prop']
non_zero_weights = weights[weights.Prop != 0]
print(non_zero_weights)
non_zero_weights.to_csv(r'C:\Users\Jonathan\Desktop\stash_weights.csv')
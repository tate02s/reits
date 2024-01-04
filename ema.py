import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from empiricaldist import Pmf
from collections import defaultdict
import re


class Price_Data:
    def __init__(self, stock, start_date="2022-01-01", end_date="2024-01-30"):
        self.stock = stock
        self.stock_df = yf.Ticker(stock).history(interval="1d", start=start_date, end=end_date)

        #   Mark a share price increase with 1 & decrease with 0
        self.stock_df["Change"] = self.stock_df["Close"].transform(lambda x: np.sign(x.diff()))
    
    def ema(self, days=14):
        self.stock_df["Ema"] = self.stock_df["Close"].transform(lambda x: x.ewm(span=days, adjust=True).mean())
    
    def calRSI(self, daysInterval=6):
        """Calculates the RSI for the days interval provided"""

        log_returns = np.log(self.stock_df["Close"]/self.stock_df["Close"].shift()).dropna()
            
        positive = log_returns.copy()
        negative = log_returns.copy()

        positive[positive < 0] = 0
        negative[negative > 0] = 0

        days = daysInterval

        averageGain = positive.rolling(window=days).mean()
        averageLoss = abs(negative.rolling(window=days).mean())

        # print(averageGain, averageLoss)

        relativeStrength = averageGain / averageLoss
        relativeStrength[relativeStrength.isna()] = 0

        RSI = 100 - (100 /(1 + relativeStrength))
        
        self.stock_df['RSI'] = RSI

    def cal_macd(self, short_days=1, long_days=2):
        short_sig = self.stock_df["Close"].transform(lambda x: x.ewm(span=short_days).mean())
        long_sig = self.stock_df["Close"].transform(lambda x: x.ewm(span=long_days).mean())

        self.stock_df["macd signal"] = short_sig - long_sig


spy = Price_Data("LL")

spy.ema(200)

plt.plot(spy.stock_df["Close"])
plt.plot(spy.stock_df["Ema"])
spy.ema(90)
plt.plot(spy.stock_df["Ema"])
plt.show()

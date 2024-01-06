import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yahoofinancials as financials
import yfinance as priceData
from collections import defaultdict

class TechnicalAnalysis:
    def __init__(self, stocks, start, end, interval='daily'):
        self.stocks = stocks
        self.start = start
        self.end = end
        self.interval = interval

    def df(self):
        stockDict = {}
        for stock in self.stocks:
            stockData = priceData.Ticker(stock).history(period=self.interval, start=self.start, end=self.end)

            # Make a columns that designates what stock the row data belongs to
            stockData['Ticker'] = stock

            # Delete the Dividend and Stock Split columns
            del stockData['Dividends']
            del stockData['Stock Splits']

            # Calculate the change in share price
            stockData['% Change'] = np.log(stockData['Close']/stockData['Close'].shift())

            # Indicate if the day's movement was an increase or decrease

            close = stockData.groupby('Ticker')['Close']
            stockData['Increase'] = close.transform(lambda x: np.sign(x.diff()))

            # Calculate the MACD Indicator
            self.calMACD(stockData, close)
            
            # Calculates the day's volitility
            self.calVolitility(stockData)

            # Calculate the RSI
            self.calRSI(stockData)

            # Calculate the Stochasic Oscillator
            self.calStochOsc(stockData)

            # Calculate Williams Oscillator
            self.calWilliamsPercR(stockData)

            # Calculate the price rate of change
            self.calPROC(stockData)

            #Calculate the On Balance Volume:

            self.calOBV(stockData)


            stockDict[stock] = stockData


        df = pd.concat([df for df in stockDict.values()])

        return df.dropna(axis=0)
    
    def calMACD(self, priceDF, close):
        ema12 = close.transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = close.transform(lambda x: x.ewm(span=26, adjust=False).mean())
        priceDF['MACD'] = ema26 - ema12
        macd = priceDF.groupby('Ticker')['MACD']
        priceDF['MACD Signal'] = macd.transform(lambda x: x.ewm(span=9, adjust=False).mean())
        priceDF['Closing 50EMA'] = close.transform(lambda x: x.ewm(com=50).mean())

        # When the MACD > MACD Signal, ie when the MACD is net positive, then a buy signal commences
        priceDF['MACD'] = priceDF['MACD'] - priceDF['MACD Signal']
        del priceDF['MACD Signal']

    def calVolitility(self, priceDF):
        """Calculates the difference between the candle tails - the difference between the open and close, with respect to the previous day's close"""
        #Define volitility as the difference between the highs - lows, with respect to the difference between the opens - closes
        priceDF['Volitility'] = ((priceDF['High'] - priceDF['Low']) - abs(priceDF['Close'] - priceDF['Open'])) / priceDF['Close'].diff()

    def calRSI(self, priceDF, daysInterval=14):
        """Calculates the RSI for the days interval provided"""

        #Calculate the RSI 
        percChange = priceDF.groupby('Ticker')['% Change']
        rsiData = []
        for symbol in percChange.groups:
            
            positive = percChange.get_group(symbol).copy()
            negative = percChange.get_group(symbol).copy()

            positive[positive < 0] = 0
            negative[negative > 0] = 0

            days = daysInterval

            averageGain = positive.rolling(window=days).mean()
            averageLoss = abs(negative.rolling(window=days).mean())

            relativeStrength = averageGain / averageLoss
            RSI = 100 - (100 /(1 + relativeStrength))
            rsiData.extend([*RSI])
        
        priceDF['RSI'] = rsiData

    
    def calStochOsc(self, priceDF, n=14):
        """Calculate the stochastic Oscillator"""

        low_14, high_14 = priceDF[['Ticker', 'Low']].copy(), priceDF[['Ticker', 'High']].copy()

        low_14 = low_14.groupby('Ticker')['Low'].transform(lambda x: x.rolling(window = n).min())
        high_14 = high_14.groupby('Ticker')['High'].transform(lambda x: x.rolling(window = n).max())

        k_percent = 100 * ((priceDF['Close'] - low_14) / (high_14 - low_14))

        priceDF['Low_14'] = low_14
        priceDF['High_14'] = high_14
        priceDF['K_Percent'] = k_percent

    def calWilliamsPercR(self, priceDF, n=14):
        """Calculate the Williams % R"""

        low_14, high_14 = priceDF[['Ticker', 'Low']].copy(), priceDF[['Ticker', 'High']].copy()

        low_14 = low_14.groupby('Ticker')['Low'].transform(lambda x: x.rolling(window=n).min())
        high_14 = high_14.groupby('Ticker')['High'].transform(lambda x: x.rolling(window=n).max())

        r_percent = ((high_14 - priceDF['Close']) / (high_14 - low_14)) * -100

        priceDF['R_Percent'] = r_percent
    
    def calPROC(self, priceDF, n=9):
        """Calculate the Price Rate of Change"""
        
        priceDF['PROC'] = priceDF.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(periods = n))

    def calOBV(self, stockData):
        """Calculate the OBV"""

        obv_values = []

        prev_obv = 0

        for i, j in zip(stockData['Close'].diff(), stockData['Volume']):

            if i > 0:
                current_obv = prev_obv + j
            elif i < 0:
                current_obv = prev_obv - j
            else:
                current_obv = prev_obv

            prev_obv = current_obv

            obv_values.append(current_obv)

        # print(pd.Series(obv_values))
        stockData['OBV'] = np.array(obv_values)
    
        
reits = TechnicalAnalysis(['COLD', 'CCI'], '2023-01-01', '2023-03-01')
reits_data = reits.df()
# print(reits_data)

#   ------------------------------------------------------------------------
# Import the required objects for RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, classification_report

X_Cols = reits_data[['RSI', 'MACD', 'OBV']]
Y_Cols = reits_data['Increase']
print(X_Cols, Y_Cols)

#Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X_Cols, Y_Cols, random_state=0)

#Create a Random Forest Classifier
"""The 'criterion' parameter of RandomForestClassifier must be a str; the optiond are: log_loss, entropy, gini"""
randForestClf = RandomForestClassifier(n_estimators = 100, oob_score = True, criterion='gini', random_state=0)
randForestClf.fit(X_train, y_train)

#Make predications

y_pred = randForestClf.predict(X_test)     
print(f'Correct Prediction (%): ', accuracy_score(y_test, randForestClf.predict(X_test), normalize = True) * 100) 

targetNames = ['Up Day', 'Down Day']

#Builds a classification report
report = classification_report(y_true=y_test, y_pred=y_pred, target_names=targetNames, output_dict = True)
reportDF = pd.DataFrame(report).transpose()
print(reportDF)

featureImportance = pd.Series(randForestClf.feature_importances_, index=X_Cols.columns).sort_values(ascending=False)

x_values = list(range(len(randForestClf.feature_importances_)))
cumulativeImportances = np.cumsum(featureImportance.values)

plt.plot(x_values, cumulativeImportances, 'g-')
plt.hlines(y=0.95, xmin=0, xmax=len(featureImportance), color='r', linestyles = 'dashed')
plt.xticks(x_values, featureImportance.index, rotation = 'vertical')

plt.xlabel('Variable')
plt.ylabel('Cumulative Importance')
plt.title('Random Forest: Feature Importance Graph')

plt.show()


#-----------------------------------------------------------------------------------------


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from empiricaldist import Pmf
from scipy.stats import ttest_1samp
from scipy.stats._kde import gaussian_kde
from collections import defaultdict

reits_data = pd.read_csv("reits_data.csv").groupby("Ticker")["Close"]

# for ticker in reits_data.groups:
#     print(f"ticker: {ticker}")
#     plt.plot([price for price in reits_data.get_group(ticker)])
#     plt.show()
        

class TrendProcessor:
    def __init__(self, closing_price_groups):
        self.groups = closing_price_groups
        self.returns_dict = {}
        self.pmf_intervals = {}
        self.trend_pmfs = defaultdict(list)
        self.grouped_intervals = defaultdict(list)

    def _gen_returns(self):
        """Generate the returns for all ticker's closing prices"""
        self.returns_dict = {}
        for ticker, closing_prices in self.groups:

            ret = np.log(closing_prices/closing_prices.shift()).dropna()
            self.returns_dict[ticker] = ret

        
    def _split_into_intervals(self, min_sample=20):

        for ticker, returns in self.returns_dict.items():
            self.returns_dict[ticker] = [round(returns[i:i+min_sample], 3) for i in range(0, len(returns), min_sample)]

    def _interval_kde(self, return_interval, return_range=(-0.1, 0.1)):
        """Return a pmf estimated by a Gaussian kde for the provided interval"""
        hypos = [x for x in np.linspace(return_range[0], return_range[1])]

        kde = gaussian_kde(return_interval)
        probs = kde.pdf(hypos)

        returns_pmf = Pmf(probs, hypos)
        returns_pmf.normalize()

        # plt.plot(returns_pmf)
        # plt.show()

        return returns_pmf
        
        
    def _create_interval_pmfs(self):
        """Create pmfs for every interval for every ticker"""
        for ticker, returns_list in self.returns_dict.items():
            pmf_intervals = []
            for return_interval in returns_list:
                pmf = self._interval_kde(return_interval)
                pmf_intervals.append(pmf)

            self.pmf_intervals[ticker] = pmf_intervals


    def _check_compatability(self, reject_thresh=0.7):
        """Test if the two samples of returns come from a different distribution."""


# t = TrendProcessor(reits_data)
# t._gen_returns()
# t._split_into_intervals()
# t._create_interval_pmfs()
# t._check_compatability()
# # t._retrieve_return_intervals()
# # print(t.grouped_intervals)
# print(t.trend_pmfs)
#----------------------------------------------------------------------------------------

COLD_CLOSINGS = reits_data.get_group("COLD")
CCI_CLOSINGS = reits_data.get_group("CCI")

class MACD:
    def __init__(self, prices, longer_days=2, shorter_days=1):
        self.prices = prices
        self.longer_days = longer_days
        self.shorter_days = shorter_days

    def ema_smoothing(self, adjust=True):
        """Create an exponential moving average for a stock"""

        longer_ema = self.prices.transform(lambda x: x.ewm(span=self.longer_days, adjust=adjust).mean())
        shorter_ema = self.prices.transform(lambda x: x.ewm(span=self.shorter_days, adjust=adjust).mean())

        return (longer_ema, shorter_ema)
    
macd = MACD(CCI_CLOSINGS)
emas = macd.ema_smoothing()

class Simulate:
    def __init__(self, prices, short_days_range, longer_days_range):
        self.prices = prices
        self.sr = short_days_range
        self.lr = longer_days_range
        self.macd_combos = []
        self.macds = []

    def calRSI(self, daysInterval=6):
        """Calculates the RSI for the days interval provided"""

        log_returns = np.log(self.stock_df[self.price_type]/self.stock_df[self.price_type].shift()).dropna()
            
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
        
        self.RSI = np.round(RSI, 2)

    def possible_macds(self):
        """long signal, short signal combos for macd"""

        for s in self.sr:
            for l in self.lr:
                pair = (l, s)
                self.macd_combos.append(pair)

    
    def create_possible_macds(self):
        
        for ldays, sdays in self.macd_combos:
            self.macds.append(MACD(self.prices, ldays, sdays))

    def simulate_holdings(self, principle=1):

        current_values = []
        returns = np.log(self.prices/self.prices.shift()).dropna()

        for macd in self.macds:
            emas = macd.ema_smoothing()
            print(f"long signal: {macd.longer_days}, short signal: {macd.shorter_days}")
            current_value = principle

            for i, return_ind in enumerate(returns.index):
                if i < len(returns):
                    #   ***When the shorter ema (stock price) is greater than the longer ema (2 day ema)***
                    if emas[1][return_ind] > emas[0][return_ind]:

                        current_value *= (1 + returns[return_ind])
            
            current_values.append(current_value)
            print(current_value)

        return current_values
    
    def buy_and_hold(self, principle = 1):
        returns = np.log(self.prices/self.prices.shift()).dropna()

        current_value = principle

        for i, return_ind in enumerate(returns.index):
            if i < len(returns):
                current_value *= (1 + returns[return_ind])
        
        print(f"Buy and hold profit: {current_value} starting at: {principle}")



    
# s = Simulate(COLD_CLOSINGS, [1], [2])
# s.possible_macds()
# s.create_possible_macds()
# profitability = s.simulate_holdings()
# bh = s.buy_and_hold()


m = MACD(COLD_CLOSINGS, 2, 1)
macd_signals = m.ema_smoothing()

# plt.plot(COLD_CLOSINGS)
# plt.plot(macd_signals[0])

# plt.show()

import yfinance as yf

prices = ["Open", "Close"]
tickers2watch = ["NAT", "SPY", "BIG", "RDFN", "FREE", "PLYM", "CWK", "LC", "AAN", "GRPN", 'GOOS', 'AAP', 'LL', 'WD', 'wrby', 'MCK', 'WOLF', 'AME']

for t in tickers2watch[:1]:
    BIG = yf.Ticker(t).history(interval="1d", start="2023-06-01", end="2024-01-30")
    BIG_CLOSINGS = BIG[prices[1]]

    s = Simulate(BIG_CLOSINGS, [1], [2])
    s.possible_macds()
    s.create_possible_macds()
    profitability = s.simulate_holdings()
    bh = s.buy_and_hold()

    BIG_M = MACD(BIG_CLOSINGS, 2, 1)
    BIG_SIGNALS = BIG_M.ema_smoothing()

    print(f"Viewing current ticker: {t}")

    plt.plot(BIG_CLOSINGS)
    plt.plot(BIG_SIGNALS[0])

    plt.show()

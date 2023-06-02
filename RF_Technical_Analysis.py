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

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""Finding the optimal paramaters for all indicator for a specific industry. back testing those stratigies with historical data. Performing Monte Carlo simulations  """

class MarketData:
    def __init__(self, tickers, start, end, period='1d'):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.period = period
    
    def getMarketData(self):
        data = []
        for ticker in self.tickers:
            pa = yf.Ticker(ticker).history(period=self.period, start=self.start, end=self.end)
            del pa['Dividends']
            del pa['Stock Splits']
            pa['Symbol'] = ticker
            data.append(pa)


        priceDF = pd.concat(data)

        close = priceDF.groupby('Symbol')['Close']
        priceDF['Prediction'] = close.transform(lambda x: np.sign(x.diff()))
        
        #Get the % change in price
        priceDF['% Change'] = close.transform(lambda x: (x - x.shift(1)) / x.shift(1))

        self.calMACD(priceDF, close)
        self.calVolitility(priceDF)
        self.calRSI(priceDF)
        self.calStochOsc(priceDF)
        self.calWilliamsPercR(priceDF)
        self.calPROC(priceDF)

        #Calculate the On Balance Volume:
        obv_groups = priceDF.groupby('Symbol').apply(self.calOBV)
        obvDF = pd.DataFrame(obv_groups, columns=['OBV']).reset_index(level=0)
        priceDF = priceDF.merge(obvDF, left_on='Symbol', right_on='Symbol')

        #Return the DF

        return priceDF.dropna()
    
    def calMACD(self, priceDF, close):
        ema12 = close.transform(lambda x: x.ewm(span=12, adjust=False).mean())
        ema26 = close.transform(lambda x: x.ewm(span=26, adjust=False).mean())
        priceDF['MACD'] = ema26 - ema12
        macd = priceDF.groupby('Symbol')['MACD']
        priceDF['MACD Signal'] = macd.transform(lambda x: x.ewm(span=9, adjust=False).mean())
        priceDF['Closing 50EMA'] = close.transform(lambda x: x.ewm(com=50).mean())

        priceDF['MACD'] = priceDF['MACD'] - priceDF['MACD Signal']
        del priceDF['MACD Signal']

    
    def calVolitility(self, priceDF):
        #Define volitility as the difference between the highs - lows, with respect to the difference between the opens - closes
        priceDF['Volitility'] = ((priceDF['High'] - priceDF['Low']) - abs(priceDF['Close'] - priceDF['Open'])) / priceDF['Close']


    def calRSI(self, priceDF):
        #Calculate the RSI 
        percChange = priceDF.groupby('Symbol')['% Change']
        rsiData = []
        for symbol in percChange.groups:

            positive = percChange.get_group(symbol).copy()
            negative = percChange.get_group(symbol).copy()

            positive[positive < 0] = 0
            negative[negative > 0] = 0

            days = 14

            averageGain = positive.rolling(window=days).mean()
            averageLoss = abs(negative.rolling(window=days).mean())

            relativeStrength = averageGain / averageLoss
            RSI = 100 - (100 /(1 + relativeStrength))
            rsiData.extend([*RSI])
        
        priceDF['RSI'] = rsiData
    
    def calStochOsc(self, priceDF):
        """Calculate the stochastic Oscillator"""
        n = 14
        low_14, high_14 = priceDF[['Symbol', 'Low']].copy(), priceDF[['Symbol', 'High']].copy()

        low_14 = low_14.groupby('Symbol')['Low'].transform(lambda x: x.rolling(window = n).min())
        high_14 = high_14.groupby('Symbol')['High'].transform(lambda x: x.rolling(window = n).max())

        k_percent = 100 * ((priceDF['Close'] - low_14) / (high_14 - low_14))

        priceDF['Low_14'] = low_14
        priceDF['High_14'] = high_14
        priceDF['K_Percent'] = k_percent

    def calWilliamsPercR(self, priceDF):
        """Calculate the Williams % R"""
        n = 14
        low_14, high_14 = priceDF[['Symbol', 'Low']].copy(), priceDF[['Symbol', 'High']].copy()

        low_14 = low_14.groupby('Symbol')['Low'].transform(lambda x: x.rolling(window=n).min())
        high_14 = high_14.groupby('Symbol')['High'].transform(lambda x: x.rolling(window=n).max())

        r_percent = ((high_14 - priceDF['Close']) / (high_14 - low_14)) * -100

        priceDF['R_Percent'] = r_percent

    def calPROC(self, priceDF):
        """Calculate the Price Rate of Change"""
        n = 9
        priceDF['PROC'] = priceDF.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods = n))
        
    def calOBV(self, group):
        """Calculate the OBV"""
        volume = group['Volume']
        change = group['Close'].diff()

        prev_obv = 0
        obv_values = []

        for i, j in zip(change, volume):

            if i > 0:
                current_obv = prev_obv + j
            elif i < 0:
                current_obv = prev_obv - j
            else:
                current_obv = prev_obv

            prev_obv = current_obv
            obv_values.append(current_obv)
        
        #Convert the symbols to another column and then perform a merge on the symbol columns for this data frame and the priceDF.

        return pd.Series(obv_values, index=group.index)
    

    def backTest(self, marketDF, startingBalance, indicators):
        testBalance = startingBalance
        ltHold = startingBalance
        buyMode = False
        # Only change the balance when Long = True
        for i, changePercent in enumerate(marketDF['% Change']):
            if buyMode == True:
                if marketDF[indicators[0]][i] >= 60 and marketDF[indicators[1]][i] < marketDF[indicators[2]][i] and marketDF['Close'][i] > marketDF[indicators[3]][i]:
                    buyMode = False
                else:
                    testBalance *= (1 + changePercent)
            else:
                if marketDF[indicators[0]][i] <= 20 and marketDF[indicators[1]][i] > marketDF[indicators[2]][i] and marketDF['Close'][i] < marketDF[indicators[3]][i]:
                    buyMode = True

            #Change the lt hold balance for the entire duration, as its never not in the market
            ltHold *= (1 + changePercent)


        return f'MACD performance {testBalance/startingBalance} lt hold performance {ltHold/startingBalance}'
    
    def getReturns(self):

        closes = self.getMarketData().groupby('Symbol')['Close']
        returns = {}
        for symbol in closes.groups:
            startVal = closes.get_group(symbol)[0]
            endVal = closes.get_group(symbol)[-1]
            print(f'symbol: {symbol} start val: {startVal}, end val: {endVal}')
            growthRate = (endVal / startVal) - 1

            returns[symbol] = growthRate
        
        return returns

        
industryDict = {
    'mortgageTickers': ['AAIC', 'ABR', 'ACRE', 'AGNC', 'AGNCL'],
    'socialMediaTickers': ['SNAP', 'META', 'PINS'],
    'shipping': ['NAT', 'TNK'],
    'prop&casInsurance': ['AFG', 'ALL', 'AXS', 'CINF'],
    'hotels': ['MAR', 'WH', 'CHH', 'HTHT', 'IHG'],
    'groceryStores': ['ACI', 'ASAI', 'KR', 'GO', 'SFM'],
    'gambling': ['ACEL', 'AGS', 'CHDN', 'DKNG', 'GAMB', 'IGT'],
}

a = MarketData(industryDict['hotels'], '2017-11-01', '2023-01-01')
stocksDF = a.getMarketData()
print(stocksDF)

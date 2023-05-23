"""class to extract statements and put them into pandas data frames from the YahooFinancials module"""

from yahoofinancials import YahooFinancials as YF
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filingTypes = ['quarterly', 'annual']

class Financials:
    def __init__(self, stocks):
        self.stocks = stocks

    def getStatements(self, statementType, frequency='annual'):
        """Gets statement (income, cash or balancesheet) data and makes a pandas data frame from the information"""

        annStatementTypeDict = {'income': 'incomeStatementHistory', 'balance': 'balanceSheetHistory', 'cash': 'cashflowStatementHistory'}
        qtrStatementTypeDict = {'income': 'incomeStatementHistoryQuarterly', 'balance': 'balanceSheetHistoryQuarterly', 'cash': 'cashflowStatementHistoryQuarterly'}
        statementsCF = {}

        # Retrieves the data and puts it into dict
        for stock in self.stocks:
            stockFin = YF(stock)
            try:
                cf = stockFin.get_financial_stmts(statement_type=statementType, frequency=frequency)
                statementsCF[stock] = cf
            except:
                print(f'financial type: {statementType} not found. Try one of these three: cash, income or balance')

        cf_df_dict = {}
        # Makes a pandas DataFrame from the cf statements retrieved for each of the companies provided
        for cfStatement in statementsCF.values():
            
            stmtTypeDict = annStatementTypeDict if frequency.lower() == 'annual' else qtrStatementTypeDict

            for stock, filingList in cfStatement[stmtTypeDict[statementType]].items():
                accItemDict = defaultdict(list)
                filingDates = set()

                for filing in filingList:
                    for filingDate, filingDict in filing.items():
                        for accItem, number in filingDict.items():
                            accItemDict[accItem].append((filingDate, number))
                            filingDates.add(filingDate)
                
                # Making a data frame by filling out the rows and columns
                df = pd.DataFrame(index=[item for item in accItemDict.keys()], columns=[date for date in filingDates])

                
                # Placing the items in their respective row/column
                for item, dateNum in accItemDict.items():
                    dateNumInds = [i for i in range(len(dateNum))]
                    for i in dateNumInds:
                        df[dateNum[i][0]][item] = dateNum[i][1]

                # Make sure the dates are chronologically sorted (least to greatest)
                cf_df_dict[stock] = df.fillna(0.0)
            
        return cf_df_dict

    
grpn = Financials(['cold'])
grpn_is = grpn.getStatements('income', filingTypes[-1])['COLD']

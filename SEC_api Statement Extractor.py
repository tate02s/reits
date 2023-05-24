"""SEC API Custom Data Structures"""
import pandas as pd
import numpy as np
import sec_api
from collections import defaultdict
import yfinance as yf

class YfFinData:
    def __init__(self, tickers, start, end, period='1d'):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.period = period

    def getShares(self):
        data = []
        for ticker in self.tickers:
            fin = yf.Ticker(ticker).get_shares_full(start=self.start, end=self.end)
            data.append(fin)
        return data

class FinData:
    def __init__(self, tickers):
        self.apiKey = '79546d5c7538e63abcd6bcc3dd8d7992ef3df1608d09afa3859246318573f657'
        self.tickerNames = tickers
        self.genInfo = None

        self.mappingApi = sec_api.MappingApi(api_key=self.apiKey)
        self.queryApi = sec_api.QueryApi(api_key=self.apiKey)
        self.xbrlApi = sec_api.XbrlApi(api_key=self.apiKey)


        self.statements  = ['StatementsOfIncome', 'StatementsOfIncomeParenthetical', 'StatementsOfComprehensiveIncome', 'StatementsOfComprehensiveIncomeParenthetical', 'BalanceSheets', 'BalanceSheetsParenthetical', 
                            'StatementsOfCashFlows', 'StatementsOfCashFlowsParenthetical', 'StatementsOfShareholdersEquity', 'StatementsOfShareholdersEquityParenthetical']

    def getSummaryDF(self):
        """Turn the general info given by sec-api into a pandas DF"""
        info = defaultdict(list)

        for ticker in self.tickerNames.split(' '):
            infoDict = self.mappingApi.resolve('ticker', ticker)
            for key, val in infoDict[0].items():
                info[key].append(val)

        self.genInfo = pd.DataFrame(data=info)

        return self.genInfo
    
    def getTypeFilings(self, ticker, startDate, endDate, formType):
        """returns the form type, accession Number and filing date of all filings that meet the parameter criteria. """

        formTypeFormat = f"\"{formType[0]}\"" if len(formType) == 1 else f'(formType:\"{formType[0]}\" OR formType:\"{formType[1]}\")'

        querySent = 'ticker:' + ticker +' AND ' + 'filedAt:{' + startDate + ' TO ' + endDate + '} AND formType:' + formTypeFormat

        query = {
        "query": { "query_string": { 
            "query": querySent
            } },
        "from": "0",
        "size": "10",
        "sort": [{ "filedAt": { "order": "desc" } }]
        }

        filings = self.queryApi.get_filings(query)

        ta_mapping = [(f['formType'], f['accessionNo'], f['filedAt']) for f in filings['filings']]

        return ta_mapping
    
    def _createStatements(self, fileStatements):
        """For every file type make a Series for the accounting item and add the value into the series if none exists for that date column"""
        #Iterate through the grouped statement types 
        statements = {}
        allowedSegmentItems = set(['CommonStockSharesIssued', 'CommonStockSharesOutstanding', 'StockRepurchasedDuringPeriodShares', 'StockIssuedDuringPeriodSharesStockOptionsExercised', 'RestructuringCharges'
                                   'CommonStockSharesAuthorized', 'CommonStockSharesAuthorized', 'StockIssuedDuringPeriodSharesNewIssues', 'CommitmentsAndContingencies '])
        for sType, statementList in fileStatements.items():
            data = defaultdict(list) #With date as the key and (accItem, value) as the value within the list
            items = set() #Holds the statement type items
            used = set()
            #Iterate through each individual statement
            for statement in statementList:
                for item, itemList in statement.items():
                    items.add(item)
                    for v in itemList:
                        if type(v) == dict:
                            if 'segment' not in v.keys() and 'value' in v.keys():
                                date = v['period']['startDate'] + '_' + v['period']['endDate'] if 'startDate' in v['period'].keys() else v['period']['instant'] 
                                val = v['value']


                                dv = f'{item}{date}{val}'
                                if dv not in used:
                                    data[date].append((item, val))
                                    used.add(dv)
                            elif 'segment' in v.keys() and item in allowedSegmentItems:
                                date = v['period']['startDate'] + '_' + v['period']['endDate'] if 'startDate' in v['period'].keys() else v['period']['instant'] 
                                val = v['value']


                                dv = f'{item}{date}{val}'
                                if dv not in used:
                                    data[date].append((item, val))
                                    used.add(dv)

            #Add whatever items are missing to the column
            DFs = []
            for date, itemValList in data.items():
            
                itemIndicies = [iv[0] for iv in itemValList]
                #add the missing items 
                missingItems = [i for i in items if i not in set(itemIndicies)]
                itemIndicies = itemIndicies + missingItems
                vals = [iv[1] for iv in itemValList] + [np.nan for _ in range(len(missingItems))]

                ser = pd.DataFrame(data=vals,columns=[date], index=itemIndicies, )
                
                DFs.append(ser)

            statementDF = None
            # percentValReq = 0.5

            for df in DFs:
                if statementDF is None:
                    statementDF = df
                else:
                    statementDF = statementDF.merge(df, left_index=True, right_index=True)
            
            # minValsReq = int(len(statementDF.index) * percentValReq)
            # statementDF = statementDF.dropna(axis='columns', thresh=minValsReq)

            statements[sType] = statementDF[~statementDF.index.duplicated(keep='first')]
        
        return statements
            
                
    def aggMultiFilings(self, accessionNo):
        """Sorts all the statements of all provided filings into their respective file type catigory"""

        #Grab the accession numbers
        #Incorporate a try, catch condition into here to catch and handle the 404 error (no XBRL data) and others!
        xbrl_json_filings = [self.xbrlApi.xbrl_to_json(accession_no=an) for an in accessionNo]

        fileStatements = defaultdict(list)
        for filing in xbrl_json_filings:
            for s in self.statements:
                if s in filing.keys():
                    fileStatements[s].append(filing[s])
        
        return self._createStatements(fileStatements)

      
pdm = FinData('CCI')
filingsNos = pdm.getTypeFilings('CCI', '2020-01-01', '2023-03-12', ['10-Q', '10-K'])
pdmAccessionNos = [an[1] for an in filingsNos]

filingStatements = pdm.aggMultiFilings(pdmAccessionNos)

print(filingStatements)

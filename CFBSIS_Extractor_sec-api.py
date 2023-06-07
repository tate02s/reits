import pandas as pd
import numpy as np
import sec_api
from collections import defaultdict
from datetime import datetime

class Statement:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.apiKey = '79546d5c7538e63abcd6bcc3dd8d7992ef3df1608d09afa3859246318573f657'
        self.mappingApi = sec_api.MappingApi(api_key=self.apiKey)
        self.queryApi = sec_api.QueryApi(api_key=self.apiKey)
        self.xbrlApi = sec_api.XbrlApi(api_key=self.apiKey)
        self.statementTypes = self.statements  = set(
            ['StatementsOfIncome', 'StatementsOfIncomeParenthetical', 'StatementsOfComprehensiveIncome', 'StatementsOfComprehensiveIncomeParenthetical', 'BalanceSheets', 'BalanceSheetsParenthetical', 
             'StatementsOfCashFlows', 'StatementsOfCashFlowsParenthetical', 'StatementsOfShareholdersEquity', 'StatementsOfShareholdersEquityParenthetical'])

    def _filingInfo(self, startDate, endDate, formType):
        """returns the form type, accession Number and filing date of all filings that meet the parameter criteria. 
            It should have 3 ways to access the filings, and the statements within the filings. If should try in the order provided by the documentation at: https://pypi.org/project/sec-api/#10-k10-q8-k-section-extractor-api
            1) HTM url
            2)
        """
        filingInfo = defaultdict(list)
        formTypeFormat = f"\"{formType[0]}\"" if len(formType) == 1 else f'(formType:\"{formType[0]}\" OR formType:\"{formType[1]}\")'

        querySent = 'ticker:' + self.ticker +' AND ' + 'filedAt:{' + startDate + ' TO ' + endDate + '} AND formType:' + formTypeFormat

        query = {
        "query": { "query_string": { 
            "query": querySent
            } },
        "from": "0",
        "size": "10",
        "sort": [{ "filedAt": { "order": "desc" } }]
        }

        filings = self.queryApi.get_filings(query)

        #   Organize all the info needed to access the filings into a dictionary
        info_of_interest = set(['id', 'accessionNo', 'cik', 'formType', 'filedAt', 'ticker', 'linkToHtml'])
        for filing in filings['filings']:
            for info in filing:
                if info in info_of_interest:
                    filingInfo[info].append(filing[info])

        return filingInfo

    def _sortDates(self, dates):
        """Sort the date columns in chronological order. Dates arg must be a set"""

        #   Find the greatest date, if there are two dates (seperated by an _)
        greatestDates = []
        for date in dates:
            splitDates = date.split('_')
            #   If there are multiple dates 
            if len(splitDates) > 1:
                date1 = datetime.strptime(splitDates[0], '%Y-%m-%d')
                date2 = datetime.strptime(splitDates[-1], '%Y-%m-%d')

                #   Choose the greater time between the two
                bigDate = date1 if date1 > date2 else date2
                #   Append the biggest date
                greatestDates.append(bigDate)
            else:
                #   If there is not two dates then convert the only date & add it to the greatestDates list
                convertedDate = datetime.strptime(date, '%Y-%m-%d')
                greatestDates.append(convertedDate)
        
        #   Sort the time stamps in chronological order (least to greatest)

        #   Map the origional indexes to their greatest value
        mappedDates = {ind:date for ind, date in enumerate(greatestDates)}
        #   Sort the date dictionary chronologically
        sortedDates = {ind:date for ind, date in sorted(mappedDates.items(), key=lambda indDate: indDate[1])}
        #   Convert the date set to a list, so it can be indexed
        dates = list(dates)
        #   Reogranize the dates according to the sorted indices
        orgDates = [dates[ind] for ind in sortedDates.keys()]

        return orgDates
    
    def _sortAccItems(self, accItems):
        """Takes a set of accounting items, sorts them and returns the sorted accounting items in the form of a list"""

    
    def _fillDF(self, df, itemDateValDict):
        """Fill the data frame with the accounting values. If no value exists, then place a 0 in the spot"""
        for date, itemValList in itemDateValDict.items():
            for itemVal in itemValList:
                item = itemVal[0]
                val = itemVal[1]

                df[date][item] = val

        return df

    def _convertToDF(self, statements):
        """Have each date be the columns and the acc items be the rows. Store the columns as a key and the value is a tuple with the acc item in the first index and the value in the second index.""" 
        
        itemDateValDict = defaultdict(list)
        totAccItems = set()
        totDates = set()

        for statement in statements:
            for accItem, itemInfo in statement.items():

                totAccItems.add(accItem)
                #   Iterate through the information of each accounting item data 
                for infoDict in itemInfo:
                    #   Ignore the segment information
                    if type(infoDict) == dict and 'segment' not in infoDict.keys() and 'value' in infoDict.keys():
                        # Compute the date & values, then map them to their respective date, only if its not segment data
                        date = f"{infoDict['period']['startDate']}_{infoDict['period']['endDate']}" if 'startDate' in infoDict['period'].keys() else infoDict['period']['instant']
                        value = round(float(infoDict['value']) * 10**int(infoDict['decimals']), 2)
                        itemDateValDict[date].append((accItem, value))

                        totDates.add(date)
        
        #   Sorts the dates set and returns it as a list
        totDates = self._sortDates(totDates)
        totAccItems = self._sortAccItems()
        df = pd.DataFrame(columns=totDates, index=[index for index in totAccItems])
        #   Fill the df with the correct values
        df = self._fillDF(df, itemDateValDict)

        return df

                            
    def retrieveStatement(self, startDate, endDate, formType, statementType):
        """Retrieves only the income, balance or cashflow statements. Can't retrieve any other content from the filings"""
        filingInfo = self._filingInfo(startDate, endDate, formType)

        statements = []
        # try:
        for htmlStatement in filingInfo['linkToHtml']:
            file = self.xbrlApi.xbrl_to_json(htm_url=htmlStatement)
            
            #   If the beginning date is before the ipo date, then grab all available statements
            if statementType in file.keys():
                statements.append(file[statementType])

        return self._convertToDF(statements)

        
        
formTypes = ('10-K', '10-Q')
statementTypes = ('StatementsOfIncome', 'BalanceSheets', 'StatementsOfCashFlows')
    
cold = Statement('COLD')
# print(cold._filingInfo('2023-01-01', '2023-04-01', formTypes))
print(cold.retrieveStatement('2015-01-01', '2023-04-01', formTypes, statementTypes[0]))

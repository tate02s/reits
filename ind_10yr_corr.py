
industry_etfs = {
    "telecom":['VOX', 'FCOM', 'NXTG', 'IXP', 'RSPC'],
    "utilities": ['XLU', 'PAVE', 'VPU', 'IGF', 'IFRA'],
    "technology": ['QQQ', 'VGT', 'XLK', 'IYW', 'SMH', 'IGV'],
    "realestate": ['VNQ', 'SCHH', 'XLRE', 'IYR', 'VNQI', 'REET'],
    "materials": ['GDX', 'GUNR', 'XLB', 'GDXJ', 'GNR', 'VAW'],
    "industrials": ['XLI', 'ITA', 'VIS', 'PPA', 'XAR', 'PHO', 'JETS', 'FXR', 'FIW', 'XTN'],
    "healthcare": ['XLV', 'VHT', 'IBB', 'XBI', 'IHI', 'IXJ'],
    "financials": ['XLF', 'VFH', 'KRE', 'FAS', 'IYF', 'BIZD', 'KIE', 'IAI', 'PSP', 'KCE'],
    "energy": ['XLE', 'VDE', 'AMLP', 'XOP', 'ICLN', 'OIH', 'MLPX', 'FCG'],
    "consumerstaples": ['XLP', 'VDC', 'IEV', 'IYK', 'EATV', 'FSTA'],
    "consumerdiscretionary": ['XLY', 'XLC', 'VCR', 'FXD', 'FDIS', 'XRT', 'PEJ']
}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("reit_ind_corr_v2.csv")
data.index = data["Date"]
del data["Date"]

data = data.replace(r'.', np.nan).dropna()

sns.heatmap(data.corr())
plt.show() 

import os

import yfinance as yf

# Pandas reads Ticker and date as data without multi_level_index=False

# data for AIQ (Global X Artificial Intelligence & Technology ETF)
aiq = yf.download('AIQ', start="2018-05-16", end="2026-11-10", interval='1d', multi_level_index=False) 
if "AIQ.csv" not in os.listdir("Datasets/"):
    aiq.to_csv("Datasets/AIQ.csv")

# data for QQQ (Invesco QQQ Trust)
qqq = yf.download('QQQ', start="2018-01-16", end="2026-11-10", interval='1d', multi_level_index=False)
if "QQQ.csv" not in os.listdir("Datasets/"):
    qqq.to_csv("Datasets/QQQ.csv")

# data for VIX (CBOE Volatility Index)
vix = yf.download('^VIX', start="2018-01-16", end="2026-11-10", interval='1d', multi_level_index=False)
if "VIX.csv" not in os.listdir("Datasets/"):
    vix.to_csv("Datasets/VIX.csv")

# data for SMH (VanEck Semiconductor ETF)
smh = yf.download('SMH', start="2018-01-16", end="2026-11-10", interval='1d', multi_level_index=False)
if "SMH.csv" not in os.listdir("Datasets/"):
    smh.to_csv("Datasets/SMH.csv")

print(
    "AIQ Overview and DataFrame Head:\n",
    aiq.head(),
    aiq.info(),
    "-------------------------------\n",
    "QQQ Overview and DataFrame Head:\n",
    qqq.head(),
    qqq.info(),
    "-------------------------------\n",
    "VIX Overview and DataFrame Head:\n",
    vix.head(),
    vix.info(),
    "-------------------------------\n",
    "SMH Overview and DataFrame Head:\n",
    smh.head(),
    smh.info()
)
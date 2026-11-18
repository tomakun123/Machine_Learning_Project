# Machine_Learning_Project

## Overview of project
We'll build our own dataset of daily data for the Global X Artificial Intelligence & Technology ETF (AIQ) from 2018 to now, using 13 features: AIQ's own price behavior, market indicators like the VIX and QQQ (Nasdaq), and an AI-sector signal (SMH). We'll then train logistic regression, an SVM, and a neural network to predict whether AIQ closes green or red the next trading day.

## Dataset

In order to actually retrieve the datasets, we are using yfinance, the API directly connecting to Yahoo Finance.

yahooFinance.py --> used to pull raw data for AIQ, QQQ, SMH, VIX
featureCreation.py --> creates features from csv datasets

### Features
**AIQ's own behavior**
1. 1-day return (today's move)
2. 5-day return (short-term momentum)
3. 20-day return (medium-term momentum)
4. 20-day volatility (std of daily returns)
5. Distance from the 50-day moving average
6. Signed up-day streak (+3 = three green days in a row, -2 = two red days in a row)
7. Overnight gap (today's open vs yesterday's close)
8. RSI (Relative Strength Index from last 14 trading days)
9. Relative volume (today's volume / 20-day average)

**Market context** 
10. QQQ 1-day return
11. VIX level
12. VIX 1-day change

**AI-sector signal** 
13. SMH 1-day return (semiconductors)

### Target
1 if AIQ's close on the next trading day is above today's close, else 0.

---

## A. Introduction

### The Dataset

**Source.** We built this dataset ourselves; it isn't a pre-packaged dataset from Kaggle or the UCI repository. All raw data comes from **Yahoo Finance**, which we access in Python through the open-source `yfinance` library (`yahooFinance.py`). We downloaded daily OHLCV (Open, High, Low, Close, Volume) price history for four tickers:

| Ticker | What it is | Why we use it |
|---|---|---|
| **AIQ** | Global X Artificial Intelligence & Technology ETF | The asset we're trying to predict |
| **QQQ** | Invesco QQQ Trust (tracks the Nasdaq-100) | Overall direction of the tech-heavy market |
| **^VIX** | CBOE Volatility Index | The market's level of fear and expected volatility |
| **SMH** | VanEck Semiconductor ETF | Signal from the AI hardware/chip sector that AI companies depend on |

Closing prices are auto-adjusted for splits and dividends, so returns calculated from them reflect real investor performance. The raw files are saved in `Datasets/` (`AIQ.csv`, `QQQ.csv`, `SMH.csv`, `VIX.csv`). `featureCreation.py` then turns them into the final modeling dataset, `Datasets/Model_Dataset.csv`.

**Size.** AIQ launched in May 2018, so its history begins on 2018-05-16 (2,132 trading days of raw data). We downloaded the other three tickers starting in January 2018 so they would be available for every AIQ date. Some features need a lookback window; the longest is the 50-day moving average. Because of that, the first ~50 rows of each feature are undefined and get dropped. We also drop the final row, because its "next day" hasn't happened yet and it has no target. The final dataset contains:

- **2,082 rows (samples)**, one per trading day, from **2018-07-26 to 2026-11-05**
- **13 numeric input features**
- **1 binary target column**
- No missing values

**Features.** Each row describes the market as it stood at the close of one trading day. The 13 features fall into three groups.

*AIQ's own price behavior (9 features):*
1. **`1D_Return`**: percent change in AIQ's close from yesterday to today. This is the most recent move.
2. **`5D_Return`**: percent change over the past 5 trading days (about one week). This captures short-term momentum.
3. **`20D_Return`**: percent change over the past 20 trading days (about one month). This captures medium-term momentum or trend.
4. **`20D_Volatility`**: standard deviation of daily returns over the past 20 days. It shows whether the market has recently been calm or turbulent.
5. **`Dist_50D_MA`**: how far today's close sits above or below its 50-day moving average, as a percentage. This measures whether the price is "stretched" relative to its recent trend.
6. **`Up_Day_Streak`**: a signed count of consecutive up or down days. For example, +3 means three green days in a row and −2 means two red days in a row. This tests whether streaks tend to continue or reverse.
7. **`Overnight_Gap`**: today's open compared with yesterday's close. It captures news or sentiment that arrived while the market was closed.
8. **`RSI`**: the 14-day Relative Strength Index, a 0–100 oscillator that compares average gains with average losses. Traders read values above 70 as "overbought" and below 30 as "oversold".
9. **`Rel_Volume`**: today's trading volume divided by its 20-day average. Unusually high volume can signal conviction behind a move.

*Broad market context (3 features):*

10. **`QQQ_1D_Return`**: the Nasdaq-100's daily return. AIQ holds many large tech stocks, so it tends to move with the broader tech market.
11. **`VIX_Level`**: the closing value of the VIX. It measures how much volatility the market expects over the next 30 days, often called the "fear index".
12. **`VIX_1D_Change`**: the daily percent change in the VIX. Sudden jumps in fear often accompany sell-offs.

*AI-sector signal (1 feature):*

13. **`SMH_1D_Return`**: the semiconductor ETF's daily return. Chipmakers such as NVIDIA supply the hardware behind AI, so their moves can lead or confirm moves in AI-focused stocks.

**Target.** `Target` is a binary label:

- **1 (green day)** if AIQ's closing price on the *next* trading day is higher than today's close
- **0 (red day)** otherwise

The classes are slightly imbalanced: **1,140 up days (54.8%)** and **942 down days (45.2%)**. Our models therefore need to beat a **54.8% accuracy baseline**, which is what you'd get by always guessing "up". We also report metrics beyond plain accuracy.

### The Machine-Learning Task

**What we are predicting.** This is a **supervised binary classification** problem. Using only information available at today's close, we predict whether AIQ will close **higher (1) or lower (0) on the next trading day**. We will train and compare three models of increasing flexibility:

- **Logistic regression**: a linear, interpretable baseline. Its coefficients show which features push the prediction toward "up" or "down".
- **Support Vector Machine (SVM)**: finds a maximum-margin decision boundary. With a kernel such as RBF, it can capture non-linear relationships between features.
- **Neural network**: a multi-layer model that can learn more complex interactions between features. For example, the effect of RSI might depend on the VIX level.

Stock data is ordered in time, so we will split it **chronologically**: we train on earlier years and test on later ones. A random shuffle would let the model "see the future" during training and make its results look better than they really are.

**Why it is interesting.**

1. **It's a hard, realistic problem.** The Efficient Market Hypothesis says that publicly available information is already reflected in prices, so next-day direction should be close to a coin flip. Daily returns are extremely noisy. A model that reliably beats the baseline, even by a few percentage points, would be meaningful. A model that can't would also tell us something about how efficient this market is.
2. **AI is one of the most important market themes of this decade.** AI-focused stocks have seen huge growth, hype cycles and sharp corrections since 2018. Studying an AI ETF lets us ask whether this sector behaves predictably and how strongly it is tied to the broader tech market (QQQ), to market fear (VIX) and to the chip supply chain (SMH).
3. **It compares different kinds of signals.** Our features combine the asset's own history (momentum, mean reversion and volume) with outside market context. That lets us test which kind of information actually carries predictive value, and whether non-linear models find patterns that a linear model misses.
4. **It spans very different market regimes.** The 2018–2026 window includes the 2018 sell-off, the COVID-19 crash and recovery in 2020, the 2022 bear market, and the AI boom that followed. A model has to generalize across all of these rather than memorize one environment.
5. **It has practical value.** Short-term direction forecasts feed into trading strategies, hedging decisions and risk management. Even a small, consistent edge, combined with careful evaluation, is what quantitative finance is built on.
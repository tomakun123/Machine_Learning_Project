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

## A. Abstract

## B. Introduction

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


## C. Exploratory & Unsupervised Analysis

### Figures ###
1. correlation_matrix.png
2. feature_distributions
3. features_by_target
4. up_rate_by_decile

#### Correlation Matrix Findings

**How to read the matrix**
Each cell is the Pearson correlation (r) between two variables, ranging from -1 to +1.

| r | Meaning |
|---|---|
| +1 | Perfect positive linear relationship --> if one goes up, the other goes up proportionally |
| 0 | No *linear* relationship (a nonlinear relationship could still exist) |
| -1 | Perfect negative linear relationship --> if one goes up, the other goes down proportionally |

- **Sign** = direction
- **size** = strength
- **Redundancy:** features with |r| > ~0.7 (positive *or* negative) carry mostly the same information --> aka multicollinearity
- **Noise threshold:** with n = 2,082 rows, any |r| < ~0.044 (2/sqrt(n)) is indistinguishable from 0
- The diagonal is always 1 (each variable vs itself, so perfect pos. lin. relationship), and the matrix is symmetric, so only the lower triangle is shown

**Feature vs feature: noticeable patterns**
1. **Same-day market move**
    - AIQ / QQQ / SMH 1-day returns: r = 0.87 to 0.93
    - All three are strongly negative with VIX 1-day change (r = -0.64 to -0.71) --> stocks up = fear down
    - Overnight gap also belongs here (r = 0.58 to 0.65), since the gap is part of the day's return
    - AIQ vs QQQ = 0.93 because their holdings overlap heavily (big tech)
2. **Medium-term trend**
    - 20-day return / Dist_50D_MA / RSI: r = 0.74 to 0.87
    - All three measure the same thing: "has the price been rising over the past few weeks?"
3. **Volatility / fear**
    - 20-day volatility vs VIX level: r = 0.68
    - Moderately negative with the trend group (r = -0.26 to -0.47) --> uptrends are calm while sell-offs are volatile

**Feature vs target (last row)**
- Every |r| <= 0.04, below the noise threshold
- No single feature has a linear relationship with next-day direction

**Decisions this led to**
- **Redundancy** --> L2 regularization in logistic regression (and/or replace AIQ's 1-day return with its excess return over QQQ, or use PCA)
- **No linear signal** --> expect results near 54.8% baseline (found in target balance) --> try nonlinear models (RBF SVM, neural net), since Pearson only detects straight-line relationships

#### Feature Distributions
**Bell-shaped vs skewed**
- Bell-shaped (symmetric) --> values cluster in the middle, thin out evenly on both sides, extremes are rare (minimal outliers)
- Skewed --> values pile up on one side with a long tail on the other
    - Right skew --> long tail to the right (a few very large values)
    - Left skew --> long tail to the left (a few very negative values)
- Rule of thumb: |skew| < 0.5 symmetric, 0.5 to 1 moderate, > 1 strong
- Note: x-axis is trimmed to the 0.5th to 99.5th percentile --> most extreme days are hidden (COVID crash / Fed meetings / tariff announcements); skew is computed on ALL data

  - **Roughly bell-shaped (|skew| < 0.5)** --> AIQ 1D / 5D / 20D return, Dist_50D_MA, RSI, QQQ 1D return, SMH 1D return
  - **Strongly right-skewed** --> Rel_Volume (4.60), VIX_Level (2.81), VIX_1D_Change (2.06), 20D_Volatility (1.81)
  - **Left-skewed by one outlier** --> Overnight_Gap (-2.00), from the -16.6% COVID gap (2020-03-16), hidden by the trimmed axis
  - **Discrete** --> Up_Day_Streak: counts, almost no 0s, each step outward ~half as common; range -8 to +14
  - **Target** --> 54.8% up / 45.2% down | baseline = 54.8%

**Why skew is a problem** --> extreme tail days dominate distances (SVM) and weights (LR, NN) --> most days get squeezed together

**Scale problem** --> std ranges from 0.0067 (20D_Volatility) to 16.3 (RSI), need to standardize data to have everything fit in same "space"

**Decisions this led to** (order: log --> clip --> standardize)
1. Log transform right-skewed features --> log(x): Rel_Volume, VIX_Level, 20D_Volatility; log1p(x): VIX_1D_Change (can be negative)
2. Clip all features at train-set 1st/99th percentiles (handles Overnight_Gap outlier without deleting the day) --> Up_Day_Streak instead clipped to +/-5
3. Standardize (z = (x - mean) / std) --> every feature mean 0, std 1; fit on train only
    --> IMPORTANT: standardizing changes scale, NOT shape, which is why steps 1-2 come first
4. No resampling --> no need to change how many examples each class has since data is relatively balanced (1140 up, 942 down), and creates the baseline for us, as well as each day being important

#### Features By Target
**How to read it**
- each panel splits days by what happened **next**: before an Up day (blue) vs before a Down day (orange); dashed = class mean
- If feature is predictive --> two curves would be separated
- t-test, p-value --> chance of seeing this big a gap between the Up and Down means from random noise alone --> p < 0.05 = significant

**Findings**
- Curves overlap almost completely in every panel --> class means are nearly identical
- All 13 p-values > 0.05 --> no feature's mean differs significantly before Up vs Down days
    - Closest: Overnight_Gap (p = 0.066) --> tells us (not confidently) that days opening with a bigger gap up were slightly more likely to be followed by a down day
- No nonlinear signal either --> unlike correlation, the full distributions would show differences in shape (not just mean) between Up and Down days, yet the curves match almost exactly. Confirms the correlation matrix's target row are all |r| <= 0.04

**Decisions this led to**
- No single feature separates the classes --> any signal must come from feature *combinations*, which opens the discussion to nonlinear models (RBF SVM, neural net) even more
- Keep all features (none stand out to keep or drop based on this figure) --> results will probabilty stay close to baseline of 54.8%

#### Up Rate By Decile
**How to read it**
1. Sort all 2,082 days by that feature's value, lowest to highest.
2. Split them into 10 equal groups called deciles, about 208 days each.
    - --> Decile 1 = the 10% of days with the lowest values.
    - --> Decile 10 = the 10% with the highest values.
3. For each group, the bar shows what % of those days were followed by an up day.

In summary, each bar is a representation of: when this feature was in a given 10% slice of its values, how often did AIQ close up the next day?

| Pattern across deciles 1 --> 10 | Meaning |
|---|---|
| **Staircase going up** | Higher values --> more Up days the next day (positive relationship) |
| **Staircase going down** | Higher values --> more Down days the next day (negative relationship) |
| **U-shape (high at both ends, low in the middle)** | Extreme values in *either* direction matter --> nonlinear, which correlation would miss |
| **Upside-down U** | Middle values matter, extremes don't --> also nonlinear |
| **Only one end differs** | Threshold effect, e.g. "only the most extreme 10% of days matter" |
| **Random bumps around the dashed line, no shape** | No relationship, just noise |

**Findings**
- Every bar stays close to the 54.8% base rate (between ~47% and ~61.5%)
- No panel shows a staircase, U-shape, or threshold effect --> bars jump up and down between neighboring deciles instead of following a pattern (makes sense since there are no real patterns in markets)
    - Faint hints (ex. RSI deciles 1 vs 10) are likely noise and can't be accounted for as solid evidence
- Confirms no single feature predicts next-day direction, linearly or nonlinearly

**Decisions this led to**
- Preprocessing unchanged --> deciles are rank-based, and log/clip/standardize preserve rank order (done in preprocessing)
  - IMPORTANT: preprocessing makes features usable but cannot create signal
- Any signal must come from feature combinations (as shown in other graphs as well) 
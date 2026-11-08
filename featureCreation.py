import pandas as pd

### AIQ's own behavior ###
aiq = pd.read_csv("Datasets/AIQ.csv", index_col=0, parse_dates=True)

# Target --> 1 if tomorrow’s close is greater than today’s, else 0
aiq['Target'] = (aiq['Close'].shift(-1) > aiq['Close']).astype(int)

# Feature 1 --> 1 day return
aiq['1D_Return'] = aiq['Close'].pct_change()

# Feature 2 --> 5 day return
aiq['5D_Return'] = aiq['Close'].pct_change(periods=5)

# Feature 3 --> 20 day return
aiq['20D_Return'] = aiq['Close'].pct_change(periods=20)

# Feature 4 --> 20 day volatility
aiq['20D_Volatility'] = aiq['Close'].pct_change().rolling(window=20).std()

# Feature 5 --> distance from 50 day moving average
aiq['Dist_50D_MA'] = (aiq['Close'] / aiq['Close'].rolling(window=50).mean()) - 1

# Feature 6 --> Signed up-day streak
sign = aiq['1D_Return'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

changed = (sign != sign.shift(1))
run_id = changed.cumsum()
aiq['Up_Day_Streak'] = (sign.groupby(run_id).cumcount() + 1) * sign


# Feature 7 --> Overnight gap (today's open vs yesterday's close)
aiq['Overnight_Gap'] = (aiq['Open']/aiq['Close'].shift(1))-1

# Feature 8 --> RSI (Relative Strength Index from last 14 trading days)
avg_gain = (aiq['Close'].diff().clip(lower=0).rolling(14).mean())
avg_loss = (aiq['Close'].diff().clip(upper=0).rolling(14).mean()) * -1
rs = avg_gain / avg_loss
aiq['RSI'] = 100 - (100 / (1 + rs))

# Feature 9 --> Relative volume (today's volume / 20-day average)
aiq['Rel_Volume'] = aiq['Volume'] / aiq['Volume'].rolling(window=20).mean()

### Market Context ###
qqq = pd.read_csv("Datasets/QQQ.csv", index_col=0, parse_dates=True)
vix = pd.read_csv("Datasets/VIX.csv", index_col=0, parse_dates=True)

# Feature 10 --> QQQ 1-day return
qqq['1D_Return'] = qqq['Close'].pct_change()

# Feature 11 --> VIX level
vix['VIX_Level'] = vix['Close']

# Feature 12 --> VIX 1-day change
vix['VIX_1D_Change'] = vix['Close'].pct_change()

### AI-sector signal ###
smh = pd.read_csv("Datasets/SMH.csv", index_col=0, parse_dates=True)

# Feature 13 --> SMH 1-day return
smh['1D_Return'] = smh['Close'].pct_change()

# Putting all features into a single DataFrame used as dataset for the model
dataset = pd.DataFrame(index=aiq.index, data=
    {
        '1D_Return': aiq['1D_Return'],
        '5D_Return': aiq['5D_Return'],
        '20D_Return': aiq['20D_Return'],
        '20D_Volatility': aiq['20D_Volatility'],
        'Dist_50D_MA': aiq['Dist_50D_MA'],
        'Up_Day_Streak': aiq['Up_Day_Streak'],
        'Overnight_Gap': aiq['Overnight_Gap'],
        'RSI': aiq['RSI'],
        'Rel_Volume': aiq['Rel_Volume'],
        
        'QQQ_1D_Return': qqq['1D_Return'], 
        'VIX_Level': vix['VIX_Level'],
        'VIX_1D_Change': vix['VIX_1D_Change'],
        'SMH_1D_Return': smh['1D_Return'],
        'Target': aiq['Target']
    }
).iloc[:-1].dropna()  # Drop the last row (no target) and any rows with NaN values

# Exporting finished dataset with features and target to CSV for model training
dataset.to_csv("Datasets/Model_Dataset.csv")
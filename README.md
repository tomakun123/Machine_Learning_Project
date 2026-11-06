# Machine_Learning_Project

## Overview of project
We'll build our own dataset of daily data for the Global X Artificial Intelligence & Technology ETF (AIQ) from 2018 to now, using 13 features: AIQ's own price behavior, market indicators like the VIX and QQQ (Nasdaq), and an AI-sector signal (SMH). We'll then train logistic regression, an SVM, and a neural network to predict whether AIQ closes green or red the next trading day.

## Dataset

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
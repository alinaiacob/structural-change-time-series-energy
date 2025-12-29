import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model



def addDatesCol(df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

def checkNanValues(df):
    return df.isna().sum()

def cleanData(df):
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    df = df[df['Year'] >= 2020]
    return df

def logReturns(df):
    df["log_returns"] = np.log(df["Close"]/df["Close"].shift(1))
    df = df.dropna()
    return df

def logVolume(df):
    df["log_volume"] = np.log(df["Volume"])
    df = df.dropna()
    return df

def volZScore(df):
    df["vol_zscore_20"] = (df["log_volume"] - df["log_volume"].rolling(20).mean())/df["log_volume"].rolling(20).std()
    df = df.dropna()
    return df

def cumsum(df, symbol):
    x =  df["log_returns"].abs()
    mu = x.mean()
    #cumsum parameters
    k = 0.5 * x.std() #drift
    h = 5 * x.std() #threshold
    cumsum = np.zeros(len(x))
    for i in range(1, len(x)):
        cumsum[i] = max(0, cumsum[i-1] + (x.iloc[i] - mu - k))
    breaks = np.where(cumsum > h)[0]
    break_dates = df.index[breaks]
    return df
    #print("break dates for ", symbol,break_dates)
    #plotForCumSum(df, symbol, cumsum, h)

def plotForCumSum(df, symbol, cumsum, h):
    plt.figure(figsize=(12,5))
    plt.plot(df.index, cumsum)
    plt.axhline(h, color="red", linestyle="--", label="Threshold")
    plt.legend()
    plt.title(f"CUSUM on Absolute Log Returns (Volatility regimes) for {symbol}")
    plt.show()

def rollingStd(df, window = 60):
    df["rolling_std"] = df["log_returns"].rolling(window).std()
    df["vol_proxy"] = df["log_returns"].abs()
    #parameters
    mu = df["vol_proxy"].mean()
    sigma = df["vol_proxy"].std()
    k = 0.5 * sigma
    h = 5 * sigma

    cusum = []
    s = 0
    for x in df["vol_proxy"]:
        s = max(0, s + (x - mu -k))
        cusum.append(s)
    df["cusum_vol"] = cusum
    df["cusum_event"] = df["cusum_vol"] > h
    return df

def plotVolatilityRollingStd(df, symbol):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["vol_proxy"], alpha=0.3, label="Log returns")
    plt.plot(df.index, df["rolling_std"], linewidth=2, label="Rolling std - 60 days" )
    plt.legend()
    plt.title(f"Volatility proxy and rolling standard deviation for {symbol}")
    plt.show()

def plotOverlayEvents(df, symbol):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["rolling_std"], label="Rolling STD")
    events = df[df["cusum_event"]]
    plt.scatter(events.index, events["rolling_std"], marker="o", label="CUSUM events")

    plt.legend()
    plt.title(f"CUSUM events over rolling volatility for {symbol}")
    plt.show()

def rollingGarch(df, symbol):
 returns = df["log_returns"].dropna()
 returns = returns * 100

 window = 500
 garch_vol = []

 for i in range(window, len(returns)):
     r_window = returns.iloc[i-window:i]

     model = arch_model(
         r_window,
         vol = "GARCH",
         p=1,
         q=1,
         mean="Zero",
         dist="normal"
     )
     res = model.fit(disp="off")
     sigma_t = np.sqrt(res.forecast(horizon=1).variance.values[-1,0])
     garch_vol.append(sigma_t)

     garch_vol = pd.Series(
         garch_vol,
         index=returns.index[window:]
     )
 df["garch_vol"] = garch_vol
 return df

def plotGarchRolling(df, symbol):
     plt.figure(figsize=(12, 6))
     plt.plot(df.index, df["rolling_std"], label="Rolling STD (60d)")
     plt.plot(df.index, df["garch_vol"], label="GARCH(1,1) volatility")
     plt.legend()
     plt.title(f"Rolling volatility: empirical vs conditional for {symbol}")
     plt.show()


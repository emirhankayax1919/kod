import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Verileri çek
ticker = "AAPL"
ohlcv = yf.download(ticker, "2010-01-01", "2023-03-11")

# DataFrame'i oluştur
df = pd.DataFrame(ohlcv)

# Hareketli ortalamaları ekle
def add_ma(data, periods=[10, 20, 50]):
    for period in periods:
        data[f"MA_{period}"] = data["Close"].rolling(period).mean()
    return data

df = add_ma(df, periods=[10, 20, 50])

# RSI ekleyin
def add_rsi(data, periods=[14]):
    for period in periods:
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        data[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return data

df = add_rsi(df, periods=[14])

# MACD ekleyin
def add_macd(data, period_long=26, period_short=12, period_signal=9):
    exp1 = data["Close"].ewm(span=period_long, adjust=False).mean()
    exp2 = data["Close"].ewm(span=period_short, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=period_signal, adjust=False).mean()
    data["MACD"] = macd
    data["Signal"] = signal
    return data

df = add_macd(df)

# Stokastik osilatörü ekleyin
def add_stochastic(data, period=14, smoothing=3):
    data["C-L"] = data["Close"] - data["Low"].rolling(period).min()
    data["H-L"] = data["High"].rolling(period).max() - data["Low"].rolling(period).min()
    data["%K"] = data["C-L"] / data["H-L"] * 100
    data[f"%K_{smoothing}"] = data["%K"].rolling(smoothing).mean()
    data[f"%D_{smoothing}"] = data[f"%K_{smoothing}"].rolling(smoothing).mean()
    return data.drop(columns=["C-L", "H-L"])

df = add_stochastic(df)

# Bollinger Bantları ekleyin
def add_bollinger_bands(data, periods=[20]):
    for period in periods:
        data[f"SMA_{period}"] = data["Close"].rolling(period).mean()
        data[f"STD_{period}"] = data["Close"].rolling(period).std()
        data[f"UpperBand_{period}"] = data[f"SMA_{period}"] + (data[f"STD_{period}"] * 2)
        data[f"LowerBand_{period}"] = data[f"SMA_{period}"] - (data[f"STD_{period}"] * 2)
    return data


df = add_ma(df, periods=[10, 20, 50])
df = add_rsi(df, periods=[14])
df = add_macd(df)
df = add_stochastic(df)
df = add_bollinger_bands(df, periods=[20])

# Son 20 satırdaki verileri görselleştirin
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(df[-20:]["Date"], df[-20:]["Close"], label="Close")
plt.plot(df[-20:]["Date"], df[-20:]["SMA_10"], label="SMA_10")
plt.plot(df[-20:]["Date"], df[-20:]["SMA_20"], label="SMA_20")
plt.plot(df[-20:]["Date"], df[-20:]["SMA_50"], label="SMA_50")
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df[-20:]["Date"], df[-20:]["RSI_14"], label="RSI (14)")
plt.plot(df[-20:]["Date"], df[-20:]["RSI_upper"], label="RSI Upper Bound")
plt.plot(df[-20:]["Date"], df[-20:]["RSI_lower"], label="RSI Lower Bound")
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df[-20:]["Date"], df[-20:]["MACD"], label="MACD")
plt.plot(df[-20:]["Date"], df[-20:]["Signal"], label="Signal Line")
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df[-20:]["Date"], df[-20:]["%K"], label="%K")
plt.plot(df[-20:]["Date"], df[-20:]["%D"], label="%D")
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(df[-20:]["Date"], df[-20:]["Close"], label="Close")
plt.plot(df[-20:]["Date"], df[-20:]["UpperBand_20"], label="Upper Band")
plt.plot(df[-20:]["Date"], df[-20:]["LowerBand_20"], label="Lower Band")
plt.legend()
plt.show()

# Son 50 günlük verileri kullanarak grafik çizdirme
plt.figure(figsize=(12,8))
plt.plot(df[-50:]["Date"], df[-50:]["Close"], label="Close")
plt.plot(df[-50:]["Date"], df[-50:]["UpperBand_20"], label="Upper Band")
plt.plot(df[-50:]["Date"], df[-50:]["LowerBand_20"], label="Lower Band")
plt.legend()
plt.show()




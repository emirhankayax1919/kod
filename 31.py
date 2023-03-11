# Gerekli kütüphaneleri yüklüyoruz
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# BIST100 hisse senedi verilerini yüklüyoruz
bist100 = yf.download('^XU100', start='2010-01-01')

# Verileri ön işleme tabi tutuyoruz
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bist100['Adj Close'].values.reshape(-1, 1))

# Verileri eğitim ve test veri setleri olarak ayırıyoruz
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]


# Eğitim veri setini LSTM modeline uygun hale getiriyoruz
def create_dataset(dataset, time_step=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        data_X.append(a)
        data_Y.append(dataset[i + time_step, 0])
    return np.array(data_X), np.array(data_Y)


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# LSTM modelini oluşturuyoruz
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Modeli derleyip eğitiyoruz
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Test veri setinde tahmin yaparak gerçekle karşılaştırıyoruz
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Tahmin sonuçlarını ve gerçek verileri görselleştiriyoruz
train = bist100[:int(len(scaled_data) * 0.8)]
valid = bist100[int(len(scaled_data) * 0.8):]
valid['Predictions'] = y_pred
plt.figure(figsize=(16, 8))
plt.title('LSTM Modeli Tahminleri')
plt.xlabel('Tarih')
plt.ylabel('Fiyat')
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Eğitim Veri Seti', 'Gerçek Veriler', 'Tahminler'], loc='lower right')
plt.show()

# Teknik analiz göstergeleri hesaplanıyor
def calculate_indicators(data):
    # Hareketli ortalama hesaplama
    data['sma20'] = talib.SMA(data['Close'], timeperiod=20)
    data['sma50'] = talib.SMA(data['Close'], timeperiod=50)

    # MACD hesaplama
    macd, signal, hist = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['macd'] = macd
    data['macd_signal'] = signal

    # RSI hesaplama
    data['rsi'] = talib.RSI(data['Close'], timeperiod=14)

    return data

# Veri setini yükleyin ve özellikleri hesaplayın
df = load_data('PETKM.IS', '2018-01-01', '2021-09-01')
df = calculate_features(df)
df = calculate_indicators(df)

# Girdi verilerini hazırlayın
X = df.drop(['Close', 'Date'], axis=1).values
y = df['Close'].values

# Verileri standartlaştırın
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Eğitim ve test verilerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Modeli oluşturun ve eğitin
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Tahminleri yapın ve sonuçları çizin
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

plt.plot(y_train)
plt.plot(train_predictions)
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test)
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), test_predictions)
plt.legend(['Train', 'Train Predictions', 'Test', 'Test Predictions'])
plt.show()

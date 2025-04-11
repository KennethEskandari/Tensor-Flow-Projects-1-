#Import Libraries 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading Data 
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Preprocessing Data
data = data[['Close']] #Basically telling the computer to only use the Close column

#Normalizing Data
scaler = MinMaxScaler() #This will scale the data to a range of 0 to 1
scaled_data = scaler.fit_transform(data) #This will fit the scaler to the data and transform it

#Creating a window 
window_size = 60

#Creating Sequences
x = []
y = []

for i in range(window_size, len(scaled_data)):
    x.append(scaled_data[i-window_size:i, 0]) #This will append the last 60 days of data to the x list
    y.append(scaled_data[i, 0]) #This will append the current day to the y list

x = np.array(x)
y = np.array(y)

#Creating Train and Test Sets
split = int(.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

#Creating the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1)) #This will add a dense layer with 1 unit, which is the output layer

#Compiling the Model
model.compile(optimizer='adam',loss='mean_squared_error')

#Training the Model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1)

#Evaluating the Model
predicted_prices = model.predict(x_test) #Adding Predictions 

predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transford(y_test.reshape(-1, 1)) #This will inverse transform the data back to the original scale

plt.figure(figsize=(14, 5))
plt.plot(real_prices, color='red', label='Real Prices')
plt.plot(predicted_prices, color='blue', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
















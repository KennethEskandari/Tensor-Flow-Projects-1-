#Import Libraries 
import tensorflow as tf
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

x = np.reshape(x, x.shape[0], x.shape[1], 1) #This will reshape the x array to be 3D, which is required for LSTM


#Creating Train and Test Sets
split = int(.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]









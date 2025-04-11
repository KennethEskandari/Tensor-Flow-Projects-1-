#Import Libraries 
import tensorflow as tf
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading Data 
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
print(data.head())



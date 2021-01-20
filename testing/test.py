# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:16:47 2019

@author: djsma
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('data/IBM.csv')
training_set = dataset_train.iloc[:, 2:3].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []

for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train, y_train, epochs=100, batch_size=32)

dataset_test = pd.read_csv('data/IBM.csv')
real_stock_price = dataset_test.iloc[-20:,2:3].values
'''
inputs = dataset_test.iloc[-20:, 2:3].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60, 60+len(real_stock_price)):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
'''
inputs = x_train[-20:]
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color='red', label='Real')
plt.plot(predicted_stock_price, color='blue', label='Predicted')
plt.legend()
plt.show()

import numpy as np
data = [10.0, 11, 20, 16, 15]
data = np.array(data).reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
norm = scaler.fit_transform(data)
print(norm)
orig = scaler.inverse_transform(norm)
print(orig)
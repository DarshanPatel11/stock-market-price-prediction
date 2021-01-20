# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:09:44 2019

@author: djsma
"""
import numpy as np
from preprocessing import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

x_train, y_train, x_test, sc = train_test_split_lstm("IBM")

regressor = Sequential()
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 13), return_sequnce=True))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train, y_train, batch_size=32, epochs=50)

inputs = x_test

inputs = np.reshape(inputs, (len(inputs), 1, 13))
predicted = regressor.predict(inputs)
predicted = sc.inverse_transform(inputs)
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:12:49 2019

@author: djsma
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
def train_test_split_lstm(stock_name):
    path = os.getcwd() + "/data/" + stock_name + ".csv"
    train = pd.read_csv(path)
    train.dropna(how='any', axis=0)
    split = 0.8
    
    train = train.iloc[:, 2:].values
    #test = test.iloc[: 2:].values
    
    sc = MinMaxScaler()
    train = sc.fit_transform(train)
    #train = train[:int(split*len(train))]
    #test = data[int(split*len(data)):]
    x_train = train[0:len(train)-1]
    y_train = train[1:len(train)-1]
    
    x_test = x_train[-10:]
    
    x_train = np.reshape(x_train, (len(x_train), 1, 13))
    #y_train = np.reshape(y_train, (len(y_train, 1, 13)))
    return x_train, y_train, x_test, sc
#train_test_split_lstm("IBM")
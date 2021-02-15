# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:01:53 2019

@author: djsma
"""
import pandas as pd
import preprocess_data as ppd
import math
import lstm, time
import visualize as vs
import stock_data as sd
def preprocess():
    data = pd.read_csv('../data/IBM.csv')
    stocks = ppd.remove_data(data)
    #print(stocks.tail())
    stocks, sc_close = ppd.get_normalised_data(stocks)
    #print(stocks.head())
    stocks.to_csv('../data/IBM_preprocessed.csv', index=False)


    stocks = pd.read_csv('../data/IBM_preprocessed.csv')
    stocks_data = stocks.drop(['Item'], axis=1)
    stocks_data.dropna(how='any', axis=0)
    unroll_length = 50
    test_data_size = 200
    x_train, x_test, y_train, y_test = sd.train_test_split_lstm(stocks_data, prediction_time = 1, unroll_length = unroll_length, test_data_size=test_data_size)
    
    x_train = sd.unroll(x_train, unroll_length)
    x_test = sd.unroll(x_test, unroll_length)
    y_train = y_train[-x_train.shape[0]:]
    y_test = y_test[-x_test.shape[0]:]
    
    #print("x_train:", x_train.shape)
    #print("x_test:", x_test.shape)
    #print("y_train:", y_train.shape)
    #print("y_test:", y_test.shape)

def train(x_train, y_train, x_test, y_test, unroll_length = 50):
    batch_size = 32
    epochs = 10
    
    model = lstm.lstm_model(x_train.shape[-1], output_dim=unroll_length, return_sequences=True)
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    start = time.time()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.05)
    print('training_time', time.time() - start)

    prediction = model.predict(x_test, batch_size=batch_size)
    prediction = sc_close.inverse_transform(prediction)
    print(prediction)
    y_test = sc_close.inverse_transform(y_test)
    vs.plot_lstm_prediction(y_test, prediction)
    return model

def model_accuracy(model, x_train, y_train, x_test, y_test):
    trainScore = model.evaluate(x_train, y_train, verbose=0)
    print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))
    
    testScore = model.evaluate(x_test, y_test, verbose=0)
    print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
'''
def accuracy(prediction, y_test):
    true = 0
    for i in range(1, len(prediction)):
        if prediction[i-1] > prediction[i] and y_test[i-1] > prediction[i]:
            true += 1
        elif prediction[i-1] < prediction[i] and y_test[i-1] < prediction[i]:
            true += 1
    print("Trend Accuracy:", true/len(prediction)*100)
#accuracy(prediction, y_test)
'''
import datetime
import pandas as pd
import numpy as np
import time
import warnings

#import USDT_BTC data (3 months data, frequency: 5min)
USDT_BTC_file = 'USDT_BTC.csv'
df_USDT_BTC = pd.read_csv(USDT_BTC_file, index_col = 'date', parse_dates=True)

#Backfilling Missing Data & Replace zeros with previous non zero value
df_USDT_BTC.fillna(method="backfill", inplace=True)
df_USDT_BTC['volume'].replace(to_replace=0, method='ffill', inplace=True)

#set index
df_USDT_BTC.index = pd.to_datetime(df_USDT_BTC.index, unit='s')

#Pre-processing columns
df_USDT_BTC['open_log'] = np.log(df_USDT_BTC['open'])
df_USDT_BTC['high_log'] = np.log(df_USDT_BTC['high'])
df_USDT_BTC['low_log'] = np.log(df_USDT_BTC['low'])
df_USDT_BTC['close_log'] = np.log(df_USDT_BTC['close'])
df_USDT_BTC['volume_log'] = np.log(df_USDT_BTC['volume'])

#To predict close price minus open price
df_USDT_BTC['close_minus_open_log'] = df_USDT_BTC['close_log'] - df_USDT_BTC['open_log']

#Keep useful columns only
df_USDT_BTC = df_USDT_BTC[['open_log', 'high_log', 'low_log', 'close_log','volume_log', 'close_minus_open_log']]
input_column_number = df_USDT_BTC.shape[1]

#Training set = 90%, test set = 10%
Train_test_split = 0.9
n = int(len(df_USDT_BTC) * Train_test_split)
X_train = df_USDT_BTC.iloc[:n]
Y_train = df_USDT_BTC['close_minus_open_log'].iloc[:n]
print('X_train.shape =', X_train.shape)
print('Y_train.shape =', Y_train.shape)

X_test = df_USDT_BTC.iloc[n:]
Y_test = df_USDT_BTC['close_minus_open_log'].iloc[n:]

#Reshape your data either using array.reshape(-1, 1) if your data has a single feature 
#or array.reshape(1, -1) if it contains a single sample.
Y_train = np.expand_dims(Y_train, -1)
Y_test = np.expand_dims(Y_test, -1)
print('Y_train.shape =', Y_train.shape)
print('Y_test.shape =', Y_test.shape)

#Scale Data
from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

#ONLY FIT TO TRAININ DATA
X_scaler.fit(X_train)
Y_scaler.fit(Y_train)
scaled_X_train = X_scaler.transform(X_train)
scaled_X_test = X_scaler.transform(X_test)
print('scaled_X_train.shape =', scaled_X_train.shape)
print('scaled_X_test.shape =', scaled_X_test.shape)
scaled_Y_train = Y_scaler.transform(Y_train)
scaled_Y_test = Y_scaler.transform(Y_test)
print('scaled_Y_train.shape =', scaled_Y_train.shape)
print('scaled_Y_test.shape =', scaled_Y_test.shape)


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# define generator. Input:previous 1000 steps
n_input = 1000
Train_generator = TimeseriesGenerator(scaled_X_train, scaled_X_train, length=n_input, batch_size=1)
Test_generator = TimeseriesGenerator(scaled_X_test, scaled_X_test, length=n_input, batch_size=1)
print('scaled_X_train =\n', scaled_X_train)
print('scaled_X_test =\n', scaled_X_test)

df_scaled_Y_train = pd.DataFrame(scaled_Y_train)
df_scaled_Y_train.to_csv('df_scaled_Y_train.csv', index = True, header=True)
df_scaled_Y_test = pd.DataFrame(scaled_Y_test)
df_scaled_Y_test.to_csv('df_scaled_Y_test.csv', index = True, header=True)

print('len(Train_generator)=', len(Train_generator))


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
import math
import os
import tensorflow as tf
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

# basic model architecture
lookback_points = 1000

lstm_input = Input(shape=(lookback_points, input_column_number), name='lstm_input')
x = LSTM(100, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(128, name='dense_0')(x)
x = Activation('relu', name='relu_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('relu', name='output')(x)    

model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.005)
model.compile(optimizer='adam', loss='mse')


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, dpi = 300)
from IPython.display import Image
Image(filename='model.png')


print('scaled_X_train.shape =', scaled_X_train.shape)
print('scaled_X_test.shape =', scaled_X_test.shape)
seqModel = model.fit(
    Train_generator, 
    epochs = 25, 
    verbose = 1, 
    validation_data = Test_generator)

#Save training model
model.save('basic_model.h')

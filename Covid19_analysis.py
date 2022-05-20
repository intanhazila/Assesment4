# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:23:45 2022

@author: intanhazila
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.impute import KNNImputer

TRAIN_DATASET = os.path.join(os.getcwd(), 'Data', 'cases_malaysia_train.csv')
TEST_DATASET = os.path.join(os.getcwd(), 'Data', 'cases_malaysia_test.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_path', 'model.h5')
LOG_PATH = os.path.join(os.getcwd(), 'log')

#%% EDA
# Step 1) Load data
train_df = pd.read_csv(TRAIN_DATASET)
test_df = pd.read_csv(TEST_DATASET)

# Step 2) Analyse the loaded data
train_df.describe()
train_df.info()
train_df.duplicated().sum()
train_df.isnull().mean()

test_df.describe()
test_df.info()
test_df.duplicated().sum()
# there is 1 missing data in 1 column

# Step 3) Data cleaning
train_df['cases_new'] = train_df['cases_new'].str.replace(r'[^0-9a-zA-Z:,]+', '')
train_df['cases_new'] = train_df['cases_new'].replace(r'^\s*$', np.NaN, regex=True)
train_df['cases_new'] = train_df['cases_new'].astype('float')

test_df['cases_new'] = test_df['cases_new'].replace(r'[^0-9a-zA-Z:,]+', '')
test_df['cases_new'] = test_df['cases_new'].replace(r'^\s*$', np.NaN, regex=True)

imputer = KNNImputer(n_neighbors=24)
temp = train_df.drop(labels=['date'], axis=1) 
temp_date = train_df['date']
train_df_imputed = imputer.fit_transform(temp) 
train_df_imputed = pd.DataFrame(train_df_imputed.astype('int'))
train_df_clean = pd.concat((temp_date,train_df_imputed),axis=1)

# test_df
temp1 = test_df.drop(labels=['date'], axis=1) 
temp1_date = test_df['date']
test_df_imputed = imputer.fit_transform(temp1) 
test_df_imputed = pd.DataFrame(test_df_imputed.astype('int'))
test_df_clean = pd.concat((temp1_date,test_df_imputed),axis=1)

# Step 5) Feature selection
# Step 6) Data preprocessing
scaler = MinMaxScaler() 
train_df_clean = train_df_clean[0].values 
scaled_train_df = scaler.fit_transform(np.expand_dims(train_df_clean, axis=-1))
test_df_clean = test_df_clean[0].values 
scaled_test_df = scaler.fit_transform(np.expand_dims(test_df_clean, axis=-1))

window_size = 30

X_train=[]
Y_train=[]

for i in range(window_size, len(train_df)):
    X_train.append(scaled_train_df[i-window_size:i,0])
    Y_train.append(scaled_train_df[i,0])
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Testing dataset
temp = np.concatenate((scaled_train_df, scaled_test_df))
length_window = window_size+len(scaled_test_df)
temp = temp[-length_window:] 

X_test=[]
Y_test=[]

for i in range(window_size, len(temp)):
    X_test.append(temp[i-window_size:i,0])
    Y_test.append(temp[i,0])
    
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#%% Callback
log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# tensorboard callback
tensorboard_callback = TensorBoard(log_dir=log_dir)

# early stopping callback
early_stopping_callback = EarlyStopping(monitor='loss', patience=10)

#%% Model creation
model = Sequential()
model.add(LSTM(64, activation='tanh',
               return_sequences=(True),
               input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(64)) 
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse', metrics='mse')

hist = model.fit(X_train, Y_train, epochs=100, 
                 batch_size=128,
                 validation_data=(X_test,Y_test),
                 callbacks=[tensorboard_callback, early_stopping_callback])

print(hist.history.keys())

#%%

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot()


plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_mse'])
plt.plot()

#%% model deployment

predicted = [] #(1,60,1)
# model.predict(np.expand_dims(X_test[0], axis=0))

for test in X_test:
    predicted.append(model.predict(np.expand_dims(test,axis=0)))

predicted = np.array(predicted)

#%% Model Analysis

plt.figure()
plt.plot(predicted.reshape(len(predicted),1))
plt.plot(Y_test)
plt.legend(['Predicted', 'Actual'])
plt.show()

y_true = Y_test
y_pred = predicted.reshape(len(predicted),1)

mean_absolute_error(y_true, y_pred)

print(mean_absolute_error(y_true, y_pred)/sum(abs(y_true))*100)

#%% Inversion of the data
y_true = scaler.inverse_transform(np.expand_dims(Y_test,axis=-1))
y_pred = scaler.inverse_transform(predicted.reshape(len(predicted),1))

plt.figure()
plt.plot(y_pred)
plt.plot(y_true)
plt.legend(['Predicted', 'Actual'])
plt.show()

#%% 

model.save(os.path.join(os.getcwd(),'model.h5'))

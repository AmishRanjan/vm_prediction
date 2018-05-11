
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import math

from os import listdir
from os.path import isfile, join

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.metrics import mean_absolute_error


# In[52]:


def read_all_data():
    mypath = "/home/cocoa/Desktop/data_science/vm_prediction/planetlab_data/20110303/"

    docs = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    all_data = []
    for doc in docs:
        all_data.append(np.loadtxt(mypath + doc))

    return(np.asarray(all_data))


# In[67]:


def split_data(dataset):
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))
    return(train,test)


# In[77]:


def create_dataset(train_data,test_data,time_ahead):
    train_y = train_data[time_ahead:,:]
    train_x = train_data[0:-time_ahead,:]
    test_y = test_data[time_ahead:,:]
    test_x = test_data[0:-time_ahead,:]
    return(train_x,train_y,test_x,test_y)


# In[70]:


data = read_all_data()
data = data.reshape(288,1052)


# In[75]:


train, test = split_data(data)


# In[80]:


time_ahead = 1
train.shape
x_train, y_train, x_test, y_test = create_dataset(train,test,time_ahead)


# In[91]:


def create_model(x_train, y_train, x_test):
    # reshape input to be [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    print(x_train.shape)
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, x_train.shape[2])))
    model.add(Dense(x_train.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    return(trainPredict,testPredict)


# In[92]:


train_predict, test_predict = create_model(x_train,y_train,x_test)


# In[96]:


print(mean_absolute_error(y_test, test_predict))


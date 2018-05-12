
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
import math

from os import listdir
from os.path import isfile, join

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout

from sklearn.metrics import mean_absolute_error


# In[8]:


def read_all_data():
    mypath = "/home/cocoa/Desktop/data_science/vm_prediction/planetlab_data/20110303/"

    docs = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    all_data = []
    for doc in docs:
        all_data.append(np.loadtxt(mypath + doc))

    return(np.asarray(all_data))


# In[9]:


def split_data(dataset):
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))
    return(train,test)


# In[10]:


def create_dataset(train_data,test_data,time_ahead):
    train_y = train_data[time_ahead:,:]
    train_x = train_data[0:-time_ahead,:]
    test_y = test_data[time_ahead:,:]
    test_x = test_data[0:-time_ahead,:]
    return(train_x,train_y,test_x,test_y)


# In[11]:


data = read_all_data()
data = data.reshape(288,1052)


# In[12]:


train, test = split_data(data)


# In[13]:


time_ahead = 1
train.shape
x_train, y_train, x_test, y_test = create_dataset(train,test,time_ahead)


# In[90]:


def new_model(  drop=0.2):
    model = Sequential()

    # Add an input layer 
    model.add(Dense(300, activation='relu', input_shape=(1052,)))
    model.add(Dropout(drop))
    model.add(Dense(105, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(1052, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[91]:


model =new_model(0.2)
model.fit(x_train, y_train, epochs=100, verbose=2)
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)


# In[92]:


print(mean_absolute_error(y_test, testPredict))


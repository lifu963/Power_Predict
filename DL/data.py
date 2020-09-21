#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable


# In[18]:


num_features = ['Power']


# In[25]:


class Data_utility(object):
    def __init__(self,df,window_size,scaler=None):
        df = df.copy()
        df.drop(columns=['date'],inplace=True)
        if scaler is None:
            scaler,_ = self.transform(df[num_features])
        _,df[num_features] = self.transform(df[num_features],scaler)
        self.scaler = scaler
        
        df_len = len(df)
        set_size = (df_len//window_size)-1
        train_data = df[:set_size*window_size].values
        y_data = df[1:set_size*window_size+1][['Power']].values
        
        train_set = range(0,len(train_data)-window_size+1)
        batch_sum = len(train_set)
        n_features = train_data.shape[1]
        
        X = torch.zeros((batch_sum,window_size,n_features))
        Y = torch.zeros((batch_sum,window_size))
        
        for i in range(batch_sum):
            start = train_set[i]
            end = train_set[i]+48
            X[i,:,:] = torch.from_numpy(train_data[start:end,:])
            Y[i,:] = torch.from_numpy(y_data[start:end].squeeze(1))
        
        self.X = X
        self.Y = Y
        
    def transform(self,x,scaler=None):
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(x)
        y = scaler.transform(x)
        return scaler,y
    
    def inverse_transform(self,x,scaler):
        return scaler.inverse_transform(x)
    
    def get_data(self):
        return [self.X,self.Y]
    
    def get_scaler(self):
        return self.scaler
    
    def get_batches(self,inputs,targets,batch_size,shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
            
        start_idx = 0
        while(start_idx<length):
            end_idx = min(length,start_idx+batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            yield Variable(X),Variable(Y)
            start_idx += batch_size


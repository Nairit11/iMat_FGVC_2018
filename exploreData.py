#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:03:46 2018

@author: nairit-11
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pandas import DataFrame

#%matplotlib inline 

# Importing JSON data
with open("train.json") as datafile1: 
    data1 = json.load(datafile1)
with open("test.json") as datafile2: 
    data2 = json.load(datafile2)
with open("validation.json") as datafile3: 
    data3 = json.load(datafile3)
    
# Converting JSON to tabular data and preparing the data for model
my_dic_data = data1
keys= my_dic_data.keys()
dict_you_want1={'my_items1':my_dic_data['annotations']for key in keys}
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
df=pd.DataFrame(dict_you_want1)
fd = pd.DataFrame(dict_you_want2)
df2=df['my_items1'].apply(pd.Series)
fd2=fd['my_items2'].apply(pd.Series)
train_data = pd.merge(df2, fd2, on='image_id', how='outer') # Training data

my_dic_data = data3
keys= my_dic_data.keys()
dict_you_want1={'my_items1':my_dic_data['annotations']for key in keys}
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
df=pd.DataFrame(dict_you_want1)
fd = pd.DataFrame(dict_you_want2)
df2=df['my_items1'].apply(pd.Series)
fd2=fd['my_items2'].apply(pd.Series)
validation_data = pd.merge(df2, fd2, on='image_id', how='outer') # Validation data

my_dic_data = data2
keys= my_dic_data.keys()
dict_you_want2={'my_items2':my_dic_data['images']for key in keys}
fd = pd.DataFrame(dict_you_want2)
test_data=fd['my_items2'].apply(pd.Series) # Test data

train_data['url'] = train_data['url'].apply(lambda x:str(x[0]))
test_data['url'] = test_data['url'].apply(lambda x:str(x[0]))
validation_data['url'] = validation_data['url'].apply(lambda x:str(x[0]))

print("Training data size",train_data.shape)
print("test data size",test_data.shape)
print("validation data size",validation_data.shape)

train_data.head()
validation_data.head()
test_data.head()


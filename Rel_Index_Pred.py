'''
Created on Mar 7, 2019
 
@author: baumj

SAIDI SAIFI Index Predictor 
'''

import numpy as np
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import pandas as pd

year = ["2013","2014","2015","2016","2017"]
dpath = "C:/Users/baumj/Documents/UW Courses/EE 511 - Intro to Statistical Learning/Project/EIA_Data/"
fname = ["/Reliability_","/Operational_Data_"]
ext = {"2013":".xls","2014":".xls","2015":".xlsx","2016":".xlsx","2017":".xlsx"}

#save data files from each year in common format 
allfiles = {} #Create dict of file paths for data from each year
alldata = {} #Create dict of dataframes from each year data
for i in year:
    allfiles[i] = {'rel':(dpath+ i +"/Reliability_"+ i +ext[i]), \
                   'ops':(dpath+ i +"/Operational_Data_"+ i +ext[i])}
    alldata[i] = {'rel': (pd.read_excel(allfiles[i]['rel'],sheet_name="RELIABILITY_States",index_col=1,header=1)), \
                  'ops': (pd.read_excel(allfiles[i]['ops'],sheet_name="States",index_col=1,header=2))}

#Import Reliability data 

#f13_ops.loc[df13.index,'Net Generation']
model = linear_model
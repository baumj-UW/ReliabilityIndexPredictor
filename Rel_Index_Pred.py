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

#Import Reliability data 
allfiles = {} #Create dict of file paths for data from each year
alldata = {} #Create dict of dataframes from each year data
for i in year:
    allfiles[i] = {'rel':(dpath+ i +"/Reliability_"+ i +ext[i]), \
                   'ops':(dpath+ i +"/Operational_Data_"+ i +ext[i])}
    alldata[i] = {'rel': (pd.read_excel(allfiles[i]['rel'],sheet_name="RELIABILITY_States",\
                                        index_col=[1,3],header=1, na_values =['.',' '], skipfooter=1,\
                                        dtype={'Utility Number':np.int})), \
                  'ops': (pd.read_excel(allfiles[i]['ops'],sheet_name="States",na_values =['.'],\
                                        index_col=[1,3],header=2, dtype={'Utility Number':np.int, \
                                                                     'NERC Region':np.str}))}

#Clean up missing data
for i in year:
    alldata[i]['rel']['SAIDI With MED'].fillna(alldata[i]['rel']['SAIDI With MED.1'],inplace=True)
    alldata[i]['ops']['NERC Region'].fillna('Unknown',inplace=True)

'''
Set up data as vectors of features
Inputs:
   - Current year - Utility Name, NERC Region
   - Previous year - SAIDI/SAIFI reports, Net Generation, Customers, 
Results:
   - Current year - SAIDI With MED 
'''
#Create vector of features for 2014 prediction
# results 
# alldata[i]['rel']['SAIDI With MED'].fillna(alldata[i]['rel']['SAIDI With MED.1'],inplace=True)
# temp_ops = alldata['2013']['ops'].set_index(['Utility Name','State'])
# temp_ops.loc[temp['SAIDI With MED'].index,:]
# 
# alldata['2014']['rel'].loc[(slice(None),'WA'),:]
''' 
Get Baseline Predictor:
Use utility region's average SAIDI and SAIFI values from the previous year
to heuristically predict the current year's values
'''
#Get regions averages
reg_avg = {} 
regions = {}
for i in year:  #figure out how to get rid of "object" dtype
    
    regions[i] = alldata[i]['ops'].groupby(by='NERC Region')
    year_reg_avg = {}
    for region in regions[i].groups:
        region_index = regions[i].groups[region].values #indices for given region
        sample_overlap = alldata[i]['rel']['SAIDI With MED'].index.intersection(region_index)
        if(sample_overlap.size>0):
            year_reg_avg[region] = np.nanmean(alldata[i]['rel'].loc[sample_overlap,\
                                                                'SAIDI With MED'].values)
    
    reg_avg[i] = year_reg_avg.copy()   
    
# Get RMSE for heuristic prediction
heur_pred = {}
keys13 = alldata['2013']['rel']['SAIDI With MED'].index
keys14 = alldata['2014']['rel']['SAIDI With MED'].index
keys14 = keys14.intersection(alldata['2014']['ops']['NERC Region'].index)
# heur_pred.fromkeys(alldata['2013']['rel']['SAIDI With MED'].index, \
#                    reg_avg['2013'][alldata['2013']['ops'].loc[97,'NERC Region']]) #returns avg value from region
pred_arr14 = np.zeros((len(keys14)))
for (i,key) in enumerate(keys14):
    pred_arr14[i] = (reg_avg['2014'][alldata['2014']['ops'].loc[key,'NERC Region']])

# this (below) overwrites repeated keys     
#heur_pred = {key: (reg_avg['2013'][alldata['2013']['ops'].loc[key,'NERC Region']]) for key in keys }

actual = alldata['2014']['rel']['SAIDI With MED'].fillna(0)
metrics.mean_squared_error(actual.values.reshape(-1,1),pred_arr13) #393822.86967195437
metrics.mean_squared_error(actual.loc[keys14].values,pred_arr14)

#f13_ops.loc[df13.index,'Net Generation']
alldata['2013']['rel'].loc[:,'SAIDI With MED']
model = linear_model.LinearRegression()
test2 = (alldata["2014"]['rel']['SAIDI With MED'].index).intersection((alldata["2015"]['rel']['SAIDI With MED'].index))

#alldata['2013']['ops'].loc[alldata['2013']['rel'].reindex,'Net Generation']
overlap = (alldata['2013']['ops'].index).intersection(alldata['2013']['rel'].index)
#data = alldata['2013']['ops'].loc[overlap,'Net Generation'].fillna(0)  ##this should be the prev. year's data
data = alldata['2013']['ops'].loc[overlap,'Total'].fillna(0)
#actual = alldata['2013']['rel'].loc[:,'SAIDI With MED'].fillna(0)
actual = alldata['2013']['rel']['SAIDI With MED'].fillna(alldata['2013']['rel']['SAIDI With MED.1'])
actual = actual.fillna(0)
model.fit(data.values.reshape(-1,1),actual.values.reshape(-1,1))
x = np.array([0,1e8]).reshape(-1,1)
plt.plot(data.values,actual.values,'x')
plt.xscale("log")
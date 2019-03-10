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
                                        index_col=1,header=1, na_values =['.'], skipfooter=1)), \
                  'ops': (pd.read_excel(allfiles[i]['ops'],sheet_name="States",na_values =['.'],\
                                        index_col=1,header=2))}

#Clean up missing data
for i in year:
    alldata[i]['rel']['SAIDI With MED'].fillna(alldata[i]['rel']['SAIDI With MED.1'],inplace=True)

''' 
Get Baseline Predictor:
Use utility region's average SAIDI and SAIFI values from the previous year
to heuristically predict the current year's values
'''
#Get regions averages
reg_avg = {} 
for i in year:
    
    regions = alldata['2013']['ops'].groupby(by='NERC Region')
    year_reg_avg = {}
    for region in regions.groups:
        region_index = regions.groups[region].values #indices for given region
        sample_overlap = alldata['2013']['rel']['SAIDI With MED'].index.intersection(region_index)
        year_reg_avg[region] = np.nanmean(alldata['2013']['rel'].loc[sample_overlap,'SAIDI With MED'].values)
    
    reg_avg[i] = year_reg_avg.copy()   
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
actual = actual.fillna
model.fit(data.values.reshape(-1,1),actual.values.reshape(-1,1))
x = np.array([0,1e8]).reshape(-1,1)
plt.plot(data.values,actual.values,'x')
plt.xscale("log")
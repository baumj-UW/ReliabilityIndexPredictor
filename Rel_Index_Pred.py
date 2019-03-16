'''
Created on Mar 7, 2019
 
@author: baumj

SAIDI SAIFI Index Predictor 
'''

import numpy as np
from sklearn import metrics, linear_model
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import preprocessing
import pandas as pd
from sklearn.cluster.spectral import discretize

year = ["2013","2014","2015","2016","2017"]
dpath = "C:/Users/baumj/Documents/UW Courses/EE 511 - Intro to Statistical Learning/Project/EIA_Data/"
fname = ["/Reliability_","/Operational_Data_","/Distribution_Systems_"]
ext = {"2012":".xls","2013":".xls","2014":".xls","2015":".xlsx","2016":".xlsx","2017":".xlsx"}

#Import Reliability data 
allfiles = {} #Create dict of file paths for data from each year
alldata = {} #Create dict of dataframes from each year data
for i in year:
    allfiles[i] = {'rel':(dpath+ i +"/Reliability_"+ i +ext[i]), \
                   'ops':(dpath+ i +"/Operational_Data_"+ i +ext[i]),\
                   'dist_sys':(dpath + i+"/Distribution_Systems_" + i +ext[i])}
    alldata[i] = {'rel': (pd.read_excel(allfiles[i]['rel'],sheet_name="RELIABILITY_States",\
                                        index_col=[1,3],header=1, na_values =['.',' '], skipfooter=1,\
                                        dtype={'Utility Number':np.int})), \
                  'ops': (pd.read_excel(allfiles[i]['ops'],sheet_name="States",na_values =['.'],\
                                        index_col=[1,3],header=2, dtype={'Utility Number':np.int, \
                                                                     'NERC Region':np.str})), \
                  'dist_sys':(pd.read_excel(allfiles[i]['dist_sys'],sheet_name="Distribution_Systems_States",\
                                            na_values =['.',' '],index_col=[1,3],header=1,\
                                            dtype={'Utility Number':np.int}))}
#repeat for 2012 but skip rel data
i="2012"
allfiles[i] = {'ops':(dpath+ i +"/Operational_Data_"+ i +ext[i])}#\
#               'dist_sys':(dpath + i+"/Distribution_Systems_" + i +ext[i])}
alldata[i] = {'ops': (pd.read_excel(allfiles[i]['ops'], sheet_name="States",\
                                         na_values =['.'],index_col=[1,3],header=2,\
                                         dtype={'Utility Number':np.int,'NERC Region':np.str}))}# ,\
#   not avail in 2012          #'dist_sys':(pd.read_excel(allfiles[i]['dist_sys'],sheet_name="Distribution_Systems_States",\
#              #                              na_values =['.',' '],index_col=[1,3],header=1,\
#               #                             dtype={'Utility Number':np.int}))}


feat_year = ["2012","2013","2014","2015","2016"]
pred_year = ["2013","2014","2015","2016","2017"]
#Clean up missing data
# try swapping indexing with .loc to speedup
#need to add check for if the data exists
discrete_vars = ['Ownership Type','NERC Region']
for i in pred_year:
    alldata[i]['rel']['SAIDI With MED'].fillna(alldata[i]['rel']['SAIDI With MED.1'],inplace=True)
#THIS ISN'T ACTUALLY REPLACING ops data, NEED TO USE =    
for i in feat_year:    
    alldata[i]['ops'].loc[slice(None),discrete_vars].fillna('Unknown',inplace=True)

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

#Loop creates dict of cleaned up results for each year where SAIDI values exist and combines with data from ops file

clean_res = {}
cont_vars = ['Summer Peak Demand', 'Winter Peak Demand', 'Net Generation', \
             'Wholesale Power Purchases', 'Exchange Energy Received',\
             'Exchange Energy Delivered', 'Net Power Exchanged', 'Wheeled Power Received',\
             'Wheeled Power Delivered', 'Net Wheeled Power', 'Transmission by Other Losses',\
             'Total Sources', 'Retail Sales', 'Sales for Resale', 'Furnished without Charge',\
             'Consumed by Respondent without Charge', 'Total Energy Losses',\
             'Total Disposition', 'From Retail Sales', 'From Delivery Customers',\
             'From Sales for Resale', 'From Credits or Adjustments', 'From Transmission', \
             'From Other', 'Total']
#consider adding loop to check for a threshold of response rates to include feature...
#or rethink how nan values are being filled ^ model weight should account for this?
features = discrete_vars + cont_vars
#features = ['Ownership Type','NERC Region', 'Total', 'Total Sources']
for i,yr in enumerate(pred_year):
    clean_data = alldata[yr]['rel'].\
    loc[np.isfinite(alldata[yr]['rel']['SAIDI With MED']),\
        ['SAIDI With MED','SAIFI With MED']] #only keep indices samples that have results
    for feat in features:
        clean_data[feat] = alldata[feat_year[i]]['ops'].loc[clean_data.index,feat]
    clean_res[yr] = clean_data.copy()
    

#handle categories (uses code method from HW2 solutions)
#need to make a list of categories to convert

names = []
# for var in discrete_vars:
#     encoder = LabelEncoder()
#     X  = encoder.fit_transform(clean_res["2013"][var].fillna("Unknown")) #need to figure out why fillna above isn't sticking
#     vectors.append(X)
#     names += [(var+'_'+cat) for cat in encoder.classes_]

#this should just do the fit based on known categories
hotenc = OneHotEncoder(handle_unknown='ignore')
X = hotenc.fit_transform(clean_res["2013"][discrete_vars].fillna("Unknown").values)
for (i,feat) in enumerate(hotenc.categories_):
    names += [(discrete_vars[i]+'_'+cat) for cat in feat]

# df_disc = pd.DataFrame(X.todense(),columns=names,index=clean_res["2013"].index) #combine this df with clean_res
# #test = 
# combo = pd.concat([clean_res["2013"],df_disc],axis=1,join_axes=[clean_res["2013"].index])
# combo = combo.drop(discrete_vars,axis=1)    
# temp2 = hotenc.transform(clean_res["2014"][['Ownership Type','NERC Region']].fillna("Unknown").values)

for yr in clean_res:
    X = hotenc.transform(clean_res[yr][discrete_vars].fillna("Unknown").values)
    df_disc = pd.DataFrame(X.todense(),columns=names,index=clean_res[yr].index)
    combo = pd.concat([clean_res[yr],df_disc],axis=1,join_axes=[clean_res[yr].index])
    clean_res[yr] = combo.drop(discrete_vars,axis=1)   
    

# for i in enumerate(pred_year):
#     clean_data = alldata[i]['rel'].loc[np.isfinite(alldata[i]['rel']['SAIDI With MED']),\
#                                        ['SAIDI With MED','SAIFI With MED']] #only keep indices samples that have results
#     for feat in features:
#         clean_data[feat] = alldata[i]['ops'].loc[clean_data.index,feat]
#     clean_res[i] = clean_data.copy() 
#Add "features" from previous year
##clean_data[['Ownership Type','NERC Region']] = alldata[i-1]['ops'].loc[clean_data.index,slice('Ownership Type','NERC Region')].copy()

# alldata[i]['rel']['SAIDI With MED'].fillna(alldata[i]['rel']['SAIDI With MED.1'],inplace=True)
# temp_ops = alldata['2013']['ops'].set_index(['Utility Name','State'])
# temp_ops.loc[temp['SAIDI With MED'].index,:]
#  
# alldata['2014']['rel'].loc[(slice(None),'WA'),:]
'''
Super Simple Baseline Predictor: predict average value from prev. year
'''
val_actual = clean_res['2015'].loc[:,'SAIDI With MED']
prev_avg = np.average(clean_res['2014'].loc[:,'SAIDI With MED'])
worst = prev_avg*np.ones(val_actual.shape)
basic_MSE = metrics.mean_squared_error(val_actual.values.reshape(-1,1), worst)

print("Worst case MSE:",basic_MSE)

''' 
Get Baseline Predictor:
Use utility region's average SAIDI and SAIFI values from the previous year
to heuristically predict the current year's values
'''
'''
#Get regions averages --> can probably change this to use the cleaned up data version 
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
'''


'''
Run linear reg with simple predictor
Set nan values to zero
'''
data = clean_res['2013'].loc[:,'Total Sources'].fillna(0)
actual = clean_res['2013'].loc[:,'SAIDI With MED']

model = linear_model.LinearRegression()
model.fit(data.values.reshape(-1,1),actual.values.reshape(-1,1))

val_data = clean_res['2015'].loc[:,'Total Sources'].fillna(0)
val_actual = clean_res['2015'].loc[:,'SAIDI With MED']
val_pred = model.predict(val_data.values.reshape(-1,1))

basic_MSE = metrics.mean_squared_error(val_actual.values.reshape(-1,1), val_pred)
print("Simple Reg case MSE:",basic_MSE)
# 235776.6370881481

'''
Run linear reg with simple predictor v2
Set nan values to zero
'''
#features = ['Ownership Type','NERC Region', 'Total', 'Total Sources'] <-- use all features 
#combine training data sets 
data = clean_res['2013'].loc[:,['Total', 'Total Sources']].fillna(0).\
append(clean_res['2014'].loc[:,['Total', 'Total Sources']].fillna(0))
actual = clean_res['2013'].loc[:,'SAIDI With MED'].\
append(clean_res['2014'].loc[:,'SAIDI With MED'])

model = linear_model.LinearRegression(normalize=True)
model.fit(data.values,actual.values)

val_data = clean_res['2015'].loc[:,['Total', 'Total Sources']].fillna(0)
val_actual = clean_res['2015'].loc[:,'SAIDI With MED']
val_pred = model.predict(val_data.values)

basic_MSE = metrics.mean_squared_error(val_actual.values, val_pred)
print("Simple Reg case MSE:",basic_MSE)
# 235773.4963535127

'''
Convert categorical data to one-hot vectors 
and include in training set
'''
data = clean_res['2013'].iloc[:,2:].fillna(0).\
append(clean_res['2014'].iloc[:,2:].fillna(0))
actual = clean_res['2013'].loc[:,'SAIDI With MED'].\
append(clean_res['2014'].loc[:,'SAIDI With MED'])


model = linear_model.LinearRegression(normalize=True)
model.fit(data.values,actual.values)

val_data = clean_res['2015'].iloc[:,2:].fillna(0)
val_actual = clean_res['2015'].loc[:,'SAIDI With MED']
val_pred = model.predict(val_data.values)

val_MSE = metrics.mean_squared_error(val_actual.values, val_pred)
print("Validation MSE:",val_MSE)  #Validation MSE: 230590.00718307553


print("Finished yet?")
'''
Extra tests  and code

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
'''
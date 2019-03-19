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

year = ["2013","2014","2015","2016","2017"]
dpath = "C:/Users/baumj/Documents/UW Courses/EE 511 - Intro to Statistical Learning/Project/EIA_Data/"
fname = ["/Reliability_","/Operational_Data_","/Distribution_Systems_"] #not used
ext = {"2012":".xls","2013":".xls","2014":".xls","2015":".xlsx","2016":".xlsx","2017":".xlsx"}


feat_year = ["2012","2013","2014","2015","2016"]
pred_year = ["2013","2014","2015","2016","2017"]

#Import Reliability data 
# allfiles = {} #Create dict of file paths for data from each year
# alldata = {} #Create dict of dataframes from each year data
def GetFeatData(yr,dpath,ext):
    '''
    Pull data to be used as features 
    '''
    yr_files = {'ops':(dpath+ yr +"/Operational_Data_"+ yr +ext),\
                'dist_sys':(dpath + yr+"/Distribution_Systems_" + yr+ext),\
                'net_mtr':(dpath + yr + "/Net_Metering_" + yr + ext)}
    yr_data = {'ops': (pd.read_excel(yr_files['ops'],sheet_name="States",na_values =['.'],\
                                        index_col=[1,3],header=2, dtype={'Utility Number':np.int, \
                                                                     'NERC Region':np.str})), \
               'net_mtr':(pd.read_excel(yr_files['net_mtr'],sheet_name="Net_Metering_States",\
                                        na_values =['.',' '],index_col=[1,3],header=2, \
                                        dtype={'Utility Number':np.int}))}
    return yr_data


def GetPredData(yr,dpath,ext):
    '''
    Pull data to be used as the predicted value
    '''
    yr_files = {'rel':(dpath+ yr +"/Reliability_"+ yr +ext)}
    yr_data = {'rel': (pd.read_excel(yr_files['rel'],sheet_name="RELIABILITY_States",\
                                        index_col=[1,3],header=1, na_values =['.',' '], skipfooter=1,\
                                        dtype={'Utility Number':np.int}))}
    return yr_data

alldata = {key:{} for key in (feat_year + ["2017"])}
for i in feat_year:
    alldata[i].update(GetFeatData(i,dpath,ext[i]))

for i in pred_year:
    alldata[i].update(GetPredData(i,dpath,ext[i]))

# 
# for i in year:
#     allfiles[i] = {'rel':(dpath+ i +"/Reliability_"+ i +ext[i]), \
#                    'ops':(dpath+ i +"/Operational_Data_"+ i +ext[i]),\
#                    'dist_sys':(dpath + i+"/Distribution_Systems_" + i +ext[i])}
#     alldata[i] = {'rel': (pd.read_excel(allfiles[i]['rel'],sheet_name="RELIABILITY_States",\
#                                         index_col=[1,3],header=1, na_values =['.',' '], skipfooter=1,\
#                                         dtype={'Utility Number':np.int})), \
#                   'ops': (pd.read_excel(allfiles[i]['ops'],sheet_name="States",na_values =['.'],\
#                                         index_col=[1,3],header=2, dtype={'Utility Number':np.int, \
#                                                                      'NERC Region':np.str})), \
#                   'dist_sys':(pd.read_excel(allfiles[i]['dist_sys'],sheet_name="Distribution_Systems_States",\
#                                             na_values =['.',' '],index_col=[1,3],header=1,\
#                                             dtype={'Utility Number':np.int}))}
# #repeat for 2012 but skip rel data
# i="2012"
# allfiles[i] = {'ops':(dpath+ i +"/Operational_Data_"+ i +ext[i])}#\
# #               'dist_sys':(dpath + i+"/Distribution_Systems_" + i +ext[i])}
# alldata[i] = {'ops': (pd.read_excel(allfiles[i]['ops'], sheet_name="States",\
#                                          na_values =['.'],index_col=[1,3],header=2,\
#                                          dtype={'Utility Number':np.int,'NERC Region':np.str}))}# ,\
# #   not avail in 2012          #'dist_sys':(pd.read_excel(allfiles[i]['dist_sys'],sheet_name="Distribution_Systems_States",\
# #              #                              na_values =['.',' '],index_col=[1,3],header=1,\
# #               #                             dtype={'Utility Number':np.int}))}


#Clean up missing data
# try swapping indexing with .loc to speedup
#need to add check for if the data exists
discrete_vars = ['Ownership Type','NERC Region']
for i in pred_year:
    alldata[i]['rel']['SAIDI With MED'].fillna(alldata[i]['rel']['SAIDI With MED.1'],inplace=True)
    alldata[i]['rel']['SAIDI Without MED'].fillna(alldata[i]['rel']['SAIDI Without MED.1'],inplace=True)
    alldata[i]['rel']['SAIFI With MED'].fillna(alldata[i]['rel']['SAIFI With MED.1'],inplace=True)
    alldata[i]['rel']['SAIFI Without MED'].fillna(alldata[i]['rel']['SAIFI Without MED.1'],inplace=True)
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
cont_vars = {'ops':['Data Year','Summer Peak Demand', 'Winter Peak Demand', 'Net Generation', \
                    'Wholesale Power Purchases', 'Exchange Energy Received',\
                    'Exchange Energy Delivered', 'Net Power Exchanged', 'Wheeled Power Received',\
                    'Wheeled Power Delivered', 'Net Wheeled Power', 'Transmission by Other Losses',\
                    'Total Sources', 'Retail Sales', 'Sales for Resale', 'Furnished without Charge',\
                    'Consumed by Respondent without Charge', 'Total Energy Losses',\
                    'Total Disposition', 'From Retail Sales', 'From Delivery Customers',\
                    'From Sales for Resale', 'From Credits or Adjustments', 'From Transmission', \
                    'From Other', 'Total'],\
             'net_mtr':['Residential', 'Commercial', \
                        'Industrial', 'Transportation', 'Total', 'Residential.1', \
                        'Commercial.1','Industrial.1', 'Transportation.1', 'Total.1',\
                        'Residential.2','Commercial.2', 'Industrial.2', 'Transportation.2',\
                        'Total.2','Residential.3', 'Commercial.3', 'Industrial.3', 'Transportation.3',\
                        'Total.3', 'Residential.4', 'Commercial.4', 'Industrial.4','Transportation.4',\
                        'Total.4', 'Residential.5', 'Commercial.5','Industrial.5', 'Transportation.5',\
                        'Total.5', 'Residential.6','Commercial.6', 'Industrial.6', 'Transportation.6',\
                        'Total.6', 'Residential.7', 'Commercial.7', 'Industrial.7', 'Transportation.7',\
                        'Total.7', 'Residential.8', 'Commercial.8', 'Industrial.8','Transportation.8',\
                        'Total.8', 'Residential.9', 'Commercial.9','Industrial.9', 'Transportation.9',\
                        'Total.9', 'Residential.10','Commercial.10', 'Industrial.10', 'Transportation.10',\
                        'Total.10','Residential.11', 'Commercial.11', 'Industrial.11', 'Transportation.11',\
                        'Total.11']}
           
# cont_vars = ['Summer Peak Demand', 'Winter Peak Demand', 'Net Generation', \
#              'Wholesale Power Purchases', 'Exchange Energy Received',\
#              'Exchange Energy Delivered', 'Net Power Exchanged', 'Wheeled Power Received',\
#              'Wheeled Power Delivered', 'Net Wheeled Power', 'Transmission by Other Losses',\
#              'Total Sources', 'Retail Sales', 'Sales for Resale', 'Furnished without Charge',\
#              'Consumed by Respondent without Charge', 'Total Energy Losses',\
#              'Total Disposition', 'From Retail Sales', 'From Delivery Customers',\
#              'From Sales for Resale', 'From Credits or Adjustments', 'From Transmission', \
#              'From Other', 'Total']
# ['Data Year', 'Utility Name', 'Residential', 'Commercial', 'Industrial',
#        'Transportation', 'Total', 'Residential.1', 'Commercial.1',
#        'Industrial.1', 'Transportation.1', 'Total.1', 'Residential.2',
#        'Commercial.2', 'Industrial.2', 'Transportation.2', 'Total.2',
#        'Residential.3', 'Commercial.3', 'Industrial.3', 'Transportation.3',
#        'Total.3', 'Residential.4', 'Commercial.4', 'Industrial.4',
#        'Transportation.4', 'Total.4', 'Residential.5', 'Commercial.5',
#        'Industrial.5', 'Transportation.5', 'Total.5', 'Residential.6',
#        'Commercial.6', 'Industrial.6', 'Transportation.6', 'Total.6',
#        'Residential.7', 'Commercial.7', 'Industrial.7', 'Transportation.7',
#        'Total.7', 'Residential.8', 'Commercial.8', 'Industrial.8',
#        'Transportation.8', 'Total.8', 'Residential.9', 'Commercial.9',
#        'Industrial.9', 'Transportation.9', 'Total.9', 'Residential.10',
#        'Commercial.10', 'Industrial.10', 'Transportation.10', 'Total.10',
#        'Residential.11', 'Commercial.11', 'Industrial.11', 'Transportation.11',
#        'Total.11']
#consider adding loop to check for a threshold of response rates to include feature...
#or rethink how nan values are being filled ^ model weight should account for this?
features = discrete_vars + cont_vars['ops']
#features = ['Ownership Type','NERC Region', 'Total', 'Total Sources']
for i,yr in enumerate(pred_year):
    clean_data = alldata[yr]['rel'].\
    loc[np.isfinite(alldata[yr]['rel']['SAIDI With MED']),\
        ['SAIDI With MED','SAIFI With MED','SAIDI Without MED','SAIFI Without MED','Number of Customers']] #only keep indices samples that have results
    
    for feat in features:
        clean_data[feat] = alldata[feat_year[i]]['ops'].loc[clean_data.index,feat]
    for feat in cont_vars['net_mtr']:
        clean_data['net_mtr_'+feat] = alldata[feat_year[i]]['net_mtr'].loc[clean_data.index,feat]
    clean_res[yr] = clean_data.copy()
    #
    #this adds prev year SAIDI value, need to figure out how to handle 2012...
    #test['SAIDI_2013'] = clean_res['2013'].loc[clean_res['2013'].index,'SAIDI With MED']

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
disc_combo = clean_res["2013"][discrete_vars].append(clean_res["2014"][discrete_vars].\
                                                     append(clean_res["2015"][discrete_vars]))
X = hotenc.fit(disc_combo.fillna("Unknown").values)
#X = hotenc.fit_transform(clean_res["2013"][discrete_vars].fillna("Unknown").values)
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
basic_MAE = metrics.median_absolute_error(val_actual.values.reshape(-1,1), worst)
print("Worst case MSE:",basic_MSE)
print("Worst case MAE:",basic_MAE)


#quick test to check baseline for predicting the utilities prev year SAIDI
test14 = clean_res['2014'].copy()
test15 = clean_res['2015'].copy()
testcombo = pd.concat([test14,test15],axis=1,join_axes=[test14.index])
test_act = testcombo['SAIDI With MED'].values[:,0]
test_pred = testcombo['SAIDI With MED'].fillna(0).values[:,1]
metrics.mean_squared_error(test_act, test_pred) #548292.735939868
metrics.median_absolute_error(test_act, test_pred) #48.058499999999995

#plot mean/var of each year to understand underlying trends
avg_saidi_MED = {yr:np.average(clean_res[yr].loc[:,'SAIDI With MED']) for yr in pred_year}
var_saidi_MED = {yr:np.var(clean_res[yr].loc[:,'SAIDI With MED']) for yr in pred_year}
avg_saidi = {yr:np.nanmean(clean_res[yr].loc[:,'SAIDI Without MED']) for yr in pred_year}
var_saidi = {yr:np.nanvar(clean_res[yr].loc[:,'SAIDI Without MED']) for yr in pred_year}
plt.figure()
plt.errorbar(avg_saidi_MED.keys(), avg_saidi_MED.values(), \
             yerr=[avg_saidi_MED.values(),var_saidi_MED.values()], fmt='o',capthick=5,ecolor='g')
plt.title("Annual SAIDI With MED Mean and Variance")
plt.yscale("log")
plt.ylim([1e0,2e6])

plt.figure()
plt.errorbar(avg_saidi.keys(), avg_saidi.values(), \
             yerr=[avg_saidi.values(),var_saidi.values()], fmt='o',capthick=5,ecolor='g')
plt.title("Annual SAIDI without MED Mean and Variance")
plt.yscale("log")
plt.ylim([1e0,2e6])

OUTLIER = 720
lim_index = {yr:(clean_res[yr][clean_res[yr]['SAIDI With MED']<OUTLIER].index) for yr in pred_year}
avg_saidi_MED = {yr:np.average(clean_res[yr].loc[lim_index[yr],'SAIDI With MED']) for yr in pred_year}
var_saidi_MED = {yr:np.var(clean_res[yr].loc[lim_index[yr],'SAIDI With MED']) for yr in pred_year}
plt.figure()
plt.errorbar(avg_saidi_MED.keys(), avg_saidi_MED.values(), \
             yerr=[avg_saidi_MED.values(),var_saidi_MED.values()], fmt='o',capthick=5,ecolor='g')
plt.title("Limited Annual SAIDI with MED Mean and Variance")


lim_index = {yr:(clean_res[yr][clean_res[yr]['SAIDI Without MED']<OUTLIER].index) for yr in pred_year}
avg_saidi = {yr:np.average(clean_res[yr].loc[lim_index[yr],'SAIDI Without MED']) for yr in pred_year}
var_saidi = {yr:np.var(clean_res[yr].loc[lim_index[yr],'SAIDI Without MED']) for yr in pred_year}
plt.figure()
plt.errorbar(avg_saidi.keys(), avg_saidi.values(), \
             yerr=[avg_saidi.values(),var_saidi.values()], fmt='o',capthick=5,ecolor='g')
plt.title("Limited Annual SAIDI without MED Mean and Variance")
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
basic_MAE = metrics.median_absolute_error(val_actual.values.reshape(-1,1), val_pred)

print("Simple Reg case MSE:",basic_MSE)
# 235776.6370881481
print("Simple Reg case MAE:",basic_MAE)
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

Limit outliers
'''
OUTLIER = 720
lim_index = {yr:(clean_res[yr][clean_res[yr]['SAIDI With MED']<OUTLIER].index) for yr in pred_year}
# avg_saidi_MED = {yr:np.average(clean_res[yr].loc[lim_index[yr],'SAIDI With MED']) for yr in pred_year}
# var_saidi_MED = {yr:np.var(clean_res[yr].loc[lim_index[yr],'SAIDI With MED']) for yr in pred_year}

data = clean_res['2013'].loc[lim_index['2013'],:].\
append(clean_res['2014'].loc[lim_index['2014'],:])

data = data.drop(['SAIDI With MED','SAIFI With MED','SAIDI Without MED',\
                  'SAIFI Without MED'],axis='columns').fillna(0)

actual = clean_res['2013'].loc[lim_index['2013'],'SAIDI With MED'].\
append(clean_res['2014'].loc[lim_index['2014'],'SAIDI With MED'])



model = linear_model.LinearRegression(normalize=True)
model.fit(data.values,actual.values)

val_data = clean_res['2015'].loc[lim_index['2015'],:]
val_data = val_data.drop(['SAIDI With MED','SAIFI With MED','SAIDI Without MED',\
                  'SAIFI Without MED'],axis='columns').fillna(0)

val_actual = clean_res['2015'].loc[lim_index['2015'],'SAIDI With MED']
val_pred = model.predict(val_data.values)

val_MSE = metrics.mean_squared_error(val_actual.values, val_pred)
val_MAE = metrics.median_absolute_error(val_actual.values, val_pred)
print("Validation MSE:",val_MSE)  #Validation MSE: 230590.00718307553
print("Validation Med. Abs Err:",val_MAE)


'''
Lasso Model
starter code by HW2 solutions
'''
def GetError(model,data,scaler,actual):
    scaled_data = scaler.transform(data)
    preds = model.predict(scaled_data)
    mse = metrics.mean_squared_error(actual, preds)
    mae = metrics.median_absolute_error(actual, preds)
    return mse, mae

scaler = preprocessing.StandardScaler()
data_scaled = scaler.fit_transform(data)

lasso_models = []
train_errs = []
train_mae_errs = []
val_errs = []
val_mae_errs = []
#non_zeros = []
alphas = list(np.linspace(0.01,1,num=25))+list(np.linspace(1,5,num=25))\
+list(np.linspace(5,50,num=25))#+list(np.linspace(25,200,num=25))
for alpha in alphas:
    model = linear_model.Lasso(alpha=alpha, normalize=False, max_iter=1000)
    model.fit(data_scaled,actual)
#     
#     coef = model.coef_
#     num_zero = (coef == 0.0).sum()
#     non_zero = X.shape[1] - num_zero
#     non_zeros.append(non_zero)
#     
    val_mse,val_mae = GetError(model, val_data, scaler,val_actual)
    train_mse,train_mae = GetError(model, data, scaler,actual)
    
    train_errs.append(train_mse)
    train_mae_errs.append(train_mae)
    val_errs.append(val_mse)
    val_mae_errs.append(val_mae)
    lasso_models.append(model)
    print(alpha, val_mse, val_mae)
#From HW2 Solution
plt.figure()
plt.plot(alphas,train_errs,label="Train") #why does this increase...?
plt.plot(alphas,val_errs,label="Validation")
plt.title("RMSEs for Lasso")
plt.xlabel("Weight [lambda]")
plt.ylabel("RMSE (minutes)")
plt.legend()

plt.figure()
plt.plot(alphas,train_mae_errs)
plt.plot(alphas,val_mae_errs)
plt.title("Med Abs for Lasso")

#features with highest weight
print(data.columns[np.nonzero(lasso_models[33].coef_)])
# Unscaling the parameters to see which coefficients have the biggest impact
rows = []
model=lasso_models[33]
for (c, scale, mu, n) in zip(model.coef_, scaler.scale_, scaler.mean_, list(val_data.columns.values)):
    rows.append({'name': n, 'scale': scale, 'coef': c, 'mean': mu})
result = pd.DataFrame(rows)

result['x'] = result.coef / result.scale
print(result.loc[:,['coef','name']].sort_values(by='coef')[:5])
print(result.loc[:,['coef','name']].sort_values(by='coef',ascending=False)[:5])
#try repeating for Ridge model
ridge_models = []
train_errs = []
train_mae_errs = []
val_errs = []
val_mae_errs = []
#non_zeros = []
alphas = list(np.linspace(0.001,1,num=25))+list(np.linspace(1,100,num=25))+list(np.linspace(100,1000,num=25))
for alpha in alphas:
    model = linear_model.Ridge(alpha=alpha, normalize=False, max_iter=1000)
    model.fit(data_scaled,actual)
   
    val_mse,val_mae = GetError(model, val_data, scaler,val_actual)
    train_mse,train_mae = GetError(model, data, scaler,actual)
    
    train_errs.append(train_mse)
    train_mae_errs.append(train_mae)
    val_errs.append(val_mse)
    val_mae_errs.append(val_mae)
    ridge_models.append(model)
    print(alpha, val_mse, val_mae)    
    
plt.figure()
plt.plot(alphas,train_errs) #why does this increase...?
plt.plot(alphas,val_errs)
plt.title("RMSEs for Ridge")

plt.figure()
plt.plot(alphas,train_mae_errs)
plt.plot(alphas,val_mae_errs)
plt.title("Med Abs for Ridge")   
 
print("Highest weight:", data.columns[np.argmax(ridge_models[57].coef_)]) 
print("Lowest weight:", data.columns[np.argmin(ridge_models[57].coef_)])



'''
Finally run tuned model against test data
'''
model = lasso_models[33]
# 
# test_data = clean_res['2016'].iloc[:,4:].fillna(0).\
# append(clean_res['2017'].iloc[:,4:].fillna(0))
# test_actual = clean_res['2016'].loc[:,'SAIDI With MED'].\
# append(clean_res['2017'].loc[:,'SAIDI With MED'])

test16_data = clean_res['2016'].loc[lim_index['2016'],:]
test16_data = test16_data.drop(['SAIDI With MED','SAIFI With MED','SAIDI Without MED',\
                  'SAIFI Without MED'],axis='columns').fillna(0)
test16_actual = clean_res['2016'].loc[lim_index['2016'],'SAIDI With MED']

                  
test17_data = clean_res['2017'].loc[lim_index['2017'],:]
test17_data = test17_data.drop(['SAIDI With MED','SAIFI With MED','SAIDI Without MED',\
                  'SAIFI Without MED'],axis='columns').fillna(0)

test17_actual = clean_res['2017'].loc[lim_index['2017'],'SAIDI With MED']


test16_mse,test16_mae = GetError(model, test16_data, scaler,test16_actual)
test17_mse,test17_mae = GetError(model, test17_data, scaler,test17_actual)

print("2016 Test RMSE:",test16_mse)
print("2017 Test RMSE:",test17_mse)

 
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
# Covid_Prediction

Assignment for SODA-502, Fall 2020. 

```
Created on Tue Sep 22 10:00:04 2020
@author: Kafi Cifci
```
```python
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
sns.set_style("darkgrid")
import json
import folium #pip install folium
import os

from sklearn import base
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,mean_squared_error,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBClassifier,XGBRegressor
import xgboost as xgb
from hyperopt import  tpe, Trials #pip install hyperopt
from warnings import filterwarnings
filterwarnings('ignore')
from numpy import nan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

import cty_map
import xgboost_hyperopt as xgb_hyp
import grp_sen_spc
dir(grp_sen_spc)

##################### Set working directory path #####################

path_cwd = r"/SoDA 502 Covid Group Project"
os.chdir(path_cwd)
os.getcwd()

##################### read in the data from replication files #####################

json_file = r"/SoDA 502 Covid Group Project/gz_2010_us_050_00_500k.json"
with open(json_file, encoding='cp1252') as f:
    high_res_county_geo = json.loads(f.read())

final_dt = pd.read_csv('/SoDA 502 Covid Group Project/sample_file.csv')
# compare_dt = pd.read_csv('SoDA 502 Covid Group Project/compare_file.csv')

##################### read in data provided by Charles #####################

# nytimes: corona cases and deaths
nytimes = pd.read_csv('/SoDA 502 Covid Group Project/nyt covid cases and deaths.csv')
# incarceration data
incarceration = pd.read_csv('/SoDA 502 Covid Group Project/incarceration data.csv')
# opioid deaths data 
opioid_deaths = pd.read_csv('/SoDA 502 Covid Group Project/opioid deaths.csv')

##################### read in additional data we decided on #####################

# ACS 2018 (5-Year Estimates); Black or African American alone or in combination with one or more other races
black_african_american = '/SoDA 502 Covid Group Project/african_american.csv'
import chardet
with open(black_african_american, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
black_african_american = pd.read_csv(black_african_american,encoding='ISO-8859-1')

# % Hispanic (From ACS-2018-5yr estimates)
hispanic_latino = '/SoDA 502 Covid Group Project/hispanic_latino.csv'
import chardet
with open(hispanic_latino, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
hispanic_latino = pd.read_csv(hispanic_latino,encoding='ISO-8859-1')

# % of for 25 years old or older with MA or higher (From ACS-2018-5yr estimates)
higher_education = '/SoDA 502 Covid Group Project/higher_education.csv'
import chardet
with open(higher_education, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
higher_education = pd.read_csv(higher_education,encoding='ISO-8859-1')

# Presidential vote/share for 2012 and 2016 (MIT election lab)
countypres_data = pd.read_excel('/SoDA 502 Covid Group Project/countypres_data.xlsx')

##################### merge data sets #####################

# drop unnecessary variables from final_dt
final_dt = final_dt.drop(columns=['State_y', 'County_y','Confirmed'])

# rename fips before merging
nytimes.rename(columns={'fips':'FIPS'}, inplace=True)

# merge replication file and nytimes updated data 
data = pd.merge(left=final_dt, right=nytimes, on='FIPS')
data = data.drop(columns=['county', 'state']) # drop duplicate variables caused by merging

# merge 'data' with black_african_american 
black_african_american = black_african_american.drop(columns=['Geo_NAME', 'Geo_QName', 'Geo_STUSAB','Geo_SUMLEV','Geo_GEOCOMP',
                          'Geo_FILEID', 'Geo_LOGRECNO', 'Geo_US', 'Geo_STATE', 'ACS18_5yr_B01003001',
                          'ACS18_5yr_B01003001s', 'ACS18_5yr_B02009001', 'ACS18_5yr_B02009001s',
                          'Geo_COUNTY']) # drop unnecessary variables 

black_african_american.rename(columns={'Geo_FIPS':'FIPS'}, inplace=True) #change variable name to merge
data = pd.merge(left=data, right=black_african_american, on='FIPS')

# merge 'data' with hispanic_latino
hispanic_latino = hispanic_latino[['Geo_FIPS','propor_latino']] # keep only variables we need
hispanic_latino.rename(columns={'Geo_FIPS':'FIPS'}, inplace=True) #change variable name to merge
data = pd.merge(left=data, right=hispanic_latino, on='FIPS')

# merge 'data' with higher_education (propor 25years with MA or higher) 
higher_education = higher_education[['Geo_FIPS','propor_MA']] # keep only variables we need
higher_education.rename(columns={'Geo_FIPS':'FIPS'}, inplace=True) #change variable name to merge
data = pd.merge(left=data, right=higher_education, on='FIPS')

# merge 'data' with countypres_data (Presidential vote/share for 2012 and 2016) 
data = pd.merge(left=data, right=countypres_data, on='FIPS')

# convert date variable into date format
data["date"] = pd.to_datetime(data["date"])

# create another df 
data2=data.copy()

# drop columns that we don't need 
data2 = data2.drop(columns=['State_x', 'County_x', 'GEOID','Latitude','Longitude',
                          'CGPS', 'month', 'year', 'day']) # drop unnecessary variables 
data3 = data2.copy()
data3 = data3[['FIPS','date', 'cases', 'deaths']] # keep only

#### Enter '0' for cases and deaths before 1st case (useful for out-of-sample prediction)
data3 = data3.set_index(
    ['date', 'FIPS']
).unstack(
    fill_value=0
).asfreq(
    'D', fill_value=0
).stack().sort_index(level=1).reset_index()

#### make count of days 
# baseDate = datetime(1900,1,1)
# data2.numericalDates = [(d - baseDate).days for d in date]

# Covid cases on one Sept 23 
data2['DateRank'] = data2.groupby('FIPS')['date'].rank(method='dense', ascending=False)
data4 = data2[data2['DateRank'] == 1.0 ]
data4.drop('DateRank', axis=1, inplace=True)
data4= data4.drop(columns=['date','cases','deaths'])

# reconstruct final data by merging data3 and data4
final_data = pd.merge(left=data3, right=data4, on='FIPS')

# create 1 daylag and difference between coronavirus cases and deaths between a day ago and it's previous day. We drop the first day since 
# lag and difference are null in the dataset. So our data starts from day 2. 
final_data['1_step_cases'] = final_data.groupby(['FIPS'])['cases'].shift()
final_data['1_step_deaths'] = final_data.groupby(['FIPS'])['deaths'].shift()

final_data['1_step_diff_cases'] = final_data.groupby(['FIPS'])['cases'].diff()
final_data['1_step_diff_deaths'] = final_data.groupby(['FIPS'])['deaths'].diff()
final_data = final_data.dropna() # drop first day 

##################### run model for cases prediction #####################

# create training set and test sets. 
#train = final_data[(final_data['date'] > '01-21-2020') & (final_data['date'] < '07-02-2020')]
#test = final_data[(final_data['date'] > '07-01-2020') & (final_data['date'] < '09-24-2020')]

from datetime import date, timedelta
start_date = date(2020, 8, 6)
end_date = date(2020,9, 23)
delta = timedelta(days=1)

# define RMSLE
from sklearn.metrics import mean_squared_log_error
def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))

## loops through dates , but very computationally intensive. Didn't run this code
#mean_error = []
#while start_date <= end_date:
#    train = data2[data2['date'] < pd.to_datetime(start_date+delta)]
#    test = data2[data2['date'] == pd.to_datetime(start_date+delta)]
#    
#    xtr, xts = train.drop(['cases','date'], axis=1), test.drop(['cases'], axis=1)
#    ytr, yts = train['cases'].values, test['cases'].values
#    mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
#    mdl.fit(xtr, ytr)
#    p = mdl.predict(xts)
#    error = rmsle(yts, p)
#    
#    p = np.expm1(mdl.predict(xts))
#
#    error = rmsle(yts, p)
#    mean_error.append(error)
#print('Mean Error = %.5f' % np.mean(mean_error))

## instead just train with data before 8/6/2020, test on data between 8/6/2020-9/23/2020 ; use Gradient Boosted Trees
train = final_data[final_data['date'] < pd.to_datetime(start_date+delta)]
test = final_data[final_data['date'] >= pd.to_datetime(start_date+delta)]
len(train)/len(final_data) # .804
len(test)/len(final_data)  # .195

ytr, yts = train['cases'].values, test['cases'].values
xtr, xts = train.drop(['cases','date','deaths','1_step_deaths','1_step_diff_deaths'], axis=1), test.drop(['cases','date','deaths','1_step_deaths','1_step_diff_deaths'], axis=1)

mdl = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
mdl.fit(xtr, np.log1p(ytr))

p = np.expm1(mdl.predict(xts))
error = rmsle(yts, p)

##################### out-of-sample prediction for cases on October 11 #####################

# out of sample prediction for october 11 - Cases 
final_data2 = final_data.copy()
n=int(17) # predict oct 11
future_pred = n
final_data2['prediction_cases'] = final_data2['cases'].shift(-future_pred) # create another cases variable and shift by 17 days 
final_data2.dropna(inplace=True)

# create training and test set
train2 = final_data2[final_data2['date'] < pd.to_datetime(start_date+delta)]
test2 = final_data2[final_data2['date'] >= pd.to_datetime(start_date+delta)]
    
ytr2, yts2 = train2['prediction_cases'].values, test2['prediction_cases'].values
xtr2, xts2 = train2.drop(['prediction_cases','cases', 'date','deaths','1_step_deaths','1_step_diff_deaths'], axis=1), test2.drop(['cases', 'prediction_cases','date','deaths','1_step_deaths','1_step_diff_deaths'], axis=1)

mdl.fit(xtr2, np.log1p(ytr2))
p2 = np.expm1(mdl.predict(xts2))
# error = rmsle(yts2, p2)

## reconcile predictions
p2  = pd.DataFrame(p2)
p2.columns = ['predicted_cases']

df_reconcile = test2.copy()
df_reconcile = df_reconcile.drop(['prediction_cases'],axis=1)
df_reconcile= df_reconcile.reset_index(drop=True)

# merge predictions with the test set to extract cases predicted on Oct 11
df_reconcile2 = pd.merge(left=df_reconcile, right=p2, left_index=True, right_index=True)

df_reconcile2['DateRank'] = df_reconcile2.groupby('FIPS')['date'].rank(method='dense', ascending=False)
df_reconcile2 = df_reconcile2[df_reconcile2['DateRank'] == 1.0 ]
df_reconcile2.drop('DateRank', axis=1, inplace=True)

##################### run model for death prediction #####################

# create training set and test sets. 
#train = final_data[(final_data['date'] > '01-21-2020') & (final_data['date'] < '07-02-2020')]
#test = final_data[(final_data['date'] > '07-01-2020') & (final_data['date'] < '09-24-2020')]

from datetime import date, timedelta
start_date = date(2020, 8, 6)
end_date = date(2020,9, 23)
delta = timedelta(days=1)

# define RMSLE
from sklearn.metrics import mean_squared_log_error
def rmsle(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))

## loops through dates , but very computationally intensive. Didn't run this code
#mean_error = []
#while start_date <= end_date:
#    train = data2[data2['date'] < pd.to_datetime(start_date+delta)]
#    test = data2[data2['date'] == pd.to_datetime(start_date+delta)]
#    
#    xtr, xts = train.drop(['cases','date'], axis=1), test.drop(['cases'], axis=1)
#    ytr, yts = train['cases'].values, test['cases'].values
#    mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
#    mdl.fit(xtr, ytr)
#    p = mdl.predict(xts)
#    error = rmsle(yts, p)
#    
#    p = np.expm1(mdl.predict(xts))
#
#    error = rmsle(yts, p)
#    mean_error.append(error)
#print('Mean Error = %.5f' % np.mean(mean_error))

## instead just train with data before 8/6/2020, test on data between 8/6/2020-9/23/2020 ; use Gradient Boosted Trees
train3 = final_data[final_data['date'] < pd.to_datetime(start_date+delta)]
test3 = final_data[final_data['date'] >= pd.to_datetime(start_date+delta)]

ytr3, yts3 = train3['deaths'].values, test3['deaths'].values
xtr3, xts3 = train3.drop(['cases','date','deaths','1_step_cases','1_step_diff_cases'], axis=1), test3.drop(['cases','date','deaths','1_step_cases','1_step_diff_cases'], axis=1)

mdl = LGBMRegressor(n_estimators=1000, learning_rate=0.01)
mdl.fit(xtr3, np.log1p(ytr3))

p3 = np.expm1(mdl.predict(xts3))
error = rmsle(yts3, p3)

##################### out-of-sample prediction for deaths on October 11 #####################

# out of sample prediction for october 11 - Cases 
final_data2 = final_data.copy()
n=int(17) # predict oct 11
future_pred = n
final_data2['prediction_deaths'] = final_data2['deaths'].shift(-future_pred) # create another cases variable and shift by 17 days 
final_data2.dropna(inplace=True)

# create training and test set
train4 = final_data2[final_data2['date'] < pd.to_datetime(start_date+delta)]
test4 = final_data2[final_data2['date'] >= pd.to_datetime(start_date+delta)]
    
ytr4, yts4 = train4['prediction_deaths'].values, test4['prediction_deaths'].values
xtr4, xts4 = train4.drop(['prediction_deaths','cases', 'date','deaths','1_step_cases','1_step_diff_cases'], axis=1), test4.drop(['cases', 'prediction_deaths','date','deaths','1_step_cases','1_step_diff_cases'], axis=1)

mdl.fit(xtr4, np.log1p(ytr4))
p4 = np.expm1(mdl.predict(xts4))
# error = rmsle(yts2, p2)

## reconcile predictions
p4  = pd.DataFrame(p4)
p4.columns = ['predicted_deaths']

# merge predictions with the test set to extract cases predicted on Oct 11
df_reconcile3= pd.merge(left=df_reconcile, right=p4, left_index=True, right_index=True)

df_reconcile3['DateRank'] = df_reconcile3.groupby('FIPS')['date'].rank(method='dense', ascending=False)
df_reconcile3 = df_reconcile3[df_reconcile3['DateRank'] == 1.0 ]
df_reconcile3.drop('DateRank', axis=1, inplace=True)

df_reconcile3 = df_reconcile3[['FIPS','predicted_deaths']] # keep only variables we need


##################### Export data #####################
df_reconcile2= pd.merge(left=df_reconcile2, right=df_reconcile3, on='FIPS')

df_reconcile2.to_csv("Predictions.csv", sep='\t')
df_reconcile2.to_excel("Predictions.xlsx")



##################### Additional work #####################

##### our model for predicting cases without time series

data5=data.copy()
data5['DateRank'] = data5.groupby('FIPS')['date'].rank(method='dense', ascending=False)
final_data3 = data5[data5['DateRank'] == 1.0 ]
final_data3.drop('DateRank', axis=1, inplace=True)
final_data3

X_reg = final_data3[["TOT_POP","pop_den_permile","diab_perc","hyper_perc","old_perc","CRUDE_RATE","CRD_MR","Latitude","Longitude",
          "propor_black", "propor_latino", "propor_MA",
              "dem_share_2012", "rep_share_2012", "dem_share_2016", "rep_share_2016"]]
y_reg = final_data3[["cases"]]
t_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size= t_size, random_state=6886)

#define the loss function HYPEROPT class is minimizing thus negative of auc 
xgb_para = dict()
xgb_para['reg_params'] = xgb_hyp.xgb_reg_params
xgb_para['fit_params'] = xgb_hyp.xgb_fit_params_r
xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

#calling hyperopt package for regression
obj = xgb_hyp.HPOpt_reg(X_train, X_test, y_train, y_test)
result, tr  =obj.process(fn_name='xgb_reg', space=xgb_para, trials=Trials(),algo=tpe.suggest, max_evals=xgb_hyp.iterations)

final_param = {}
for k,v in result.items():
    final_param.update({k:xgb_hyp.par_va_nm[k][v]})
param = {}
for k,v in final_param.items():
    if isinstance(v, float):
        v = round(v,2)
    param.update({k:[v]})

fit_params={"early_stopping_rounds":50,            "eval_metric" : "rmse",            "eval_set" : [[X_test, y_test]]}
xgb = XGBRegressor(importance_type='weight',silent = True,verbose_eval=50)
reg = GridSearchCV(xgb,param, verbose=1,                   cv=5)
reg.fit(X_train,y_train, **fit_params)

xgb_reg = eval(str(reg.best_estimator_))
xgb_reg.fit(X_reg,y_reg)
#vi_3 = pd.DataFrame(list(zip(xgb_reg.feature_importances_,X.columns.tolist())))

#evaluating regression model
xgb_reg.fit(X_train,y_train)
print("Training Dataset")
print("RMSE:")
print(round(np.sqrt(mean_squared_error(y_train, xgb_reg.predict(X_train))),2))
print("Testing Dataset")
print("RMSE:")
print(round(np.sqrt(mean_squared_error(y_test, xgb_reg.predict(X_test))),2))





##### REVISIT - Facebook Prophet model (seasonlity, holiday, trend): 
    
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

list(data)

data = data.rename(columns={'date': 'ds', 'cases':'y'})
list(data)
from fbprophet import Prophet
grouped = data.groupby('County_x')
final = pd.DataFrame()
for g in grouped.groups:
    group = grouped.get_group(g)
    m = Prophet()
    m.fit(group)
    future = m.make_future_dataframe(periods=50)
    forecast = m.predict(future)    
    forecast = forecast.rename(columns={'yhat': 'yhat_'+g})
    final = pd.merge(final, forecast.set_index('ds'), how='outer', left_index=True, right_index=True)

final = final[['yhat_' + g for g in grouped.groups.keys()]]
##########################################################
```

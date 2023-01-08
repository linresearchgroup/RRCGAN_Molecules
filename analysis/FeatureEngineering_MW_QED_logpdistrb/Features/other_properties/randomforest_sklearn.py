# Kianoosh Sattari
# using Multi-layer Perceptron (MLP) Regressor from sklearn, nonlinear Kernels.
# Input: Features from structure, Target: Heat_Capacity
# Decide about using all samples or subsampling--> comment some parts
# 07-21-2020

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import time
import random
import pickle
import csv

start = time.time()


#reading data, NO NEED IF  have saved data, just read *.pickle
#data = pd.read_csv('Features_hc_others.csv')

data = pd.read_csv('Features_hc_others.csv')
columns = list(data.head(0))
print (columns)

X_train, X_test, y_cv_train, y_cv_test, y_E_homo_train, y_E_homo_test,y_E_lumo_train, y_E_lumo_test,y_gap_train, y_gap_test,y_r2_train, y_r2_test, y_zpve_train, y_zpve_test, y_u0_train, y_u0_test,y_u298_train, y_u298_test,y_h298_train, y_h298_test,y_g298_train, y_g298_test = train_test_split( data.iloc[:,0:18].values, data.iloc[:,18].values, data.iloc[:,19].values, data.iloc[:,20].values, data.iloc[:,21].values, data.iloc[:,22].values, data.iloc[:,23].values, data.iloc[:,24].values, data.iloc[:,25].values, data.iloc[:,26].values, data.iloc[:,27].values)

print (data.head())
print ("X_train", X_train)
print ("y_tcv_train", y_cv_train)
print ("y_zpve_train", y_zpve_train)
print ("r2: ", y_r2_train)
print ("y_u0_train", y_u0_train )
print ("g_298", y_g298_train)
with open ('features_train.pickle','wb') as f:
    pickle.dump ((X_train, y_cv_train, y_E_homo_train,y_E_lumo_train,y_gap_train,y_r2_train,y_zpve_train,y_u0_train, y_u298_train,y_h298_train,y_g298_train), f)

with open ('features_test.pickle','wb') as f:
    pickle.dump ((X_test, y_cv_test,y_E_homo_test,y_E_lumo_test,y_gap_test, y_r2_test,y_zpve_test,y_u0_test, y_u298_test,y_h298_test, y_g298_test), f)
print ("after writ, g298",y_g298_train)


with open ('features_train.pickle','rb') as f:
     X_train, y_cv_train, y_E_homo_train, y_E_lumo_train, y_gap_train, y_r2_train, y_zpve_train, y_u0_train, y_u298_train, y_h298_train, y_g298_train   = pickle.load(f)
print ("after saving, g298",y_g298_train)
with open ('features_test.pickle','rb') as f:
    X_test, y_cv_test, y_E_homo_test, y_E_lumo_test, y_gap_test, y_r2_test, y_zpve_test, y_u0_test, y_u298_test, y_h298_test , y_g298_test = pickle.load(f)

#with open ('features_subsample_train.pickle','rb') as f:
print ("g298 in the middle", y_g298_train)
#subsampling, run if you want subsampling, NO NEED IF have saved subsampling.pickle
idx = np.random.choice(len(y_cv_train) , int(len(y_cv_train)*0.25), replace = False)
X_train = X_train [idx]
y_cv_train = y_cv_train [idx]
y_E_homo_train = y_E_homo_train [idx]
y_E_lumo_train = y_E_lumo_train [idx]
y_gap_train = y_gap_train [idx]
y_r2_train = y_r2_train [idx]
y_zpve_train = y_zpve_train [idx]
y_u0_train = y_u0_train [idx]
y_u298_train = y_u298_train [idx]
y_h298_train = y_h298_train [idx]
y_g298_train = y_g298_train [idx]




idx = np.random.choice(len(y_cv_test) , int(len(y_cv_test)*0.25), replace = False)
X_test = X_test [idx]
y_cv_test = y_cv_test [idx]
y_E_homo_test = y_E_homo_test [idx]
y_E_lumo_test = y_E_lumo_test [idx]
y_gap_test = y_gap_test [idx]
y_r2_test = y_r2_test [idx]
y_zpve_test = y_zpve_test [idx]
y_u0_test = y_u0_test [idx]
y_u298_test = y_u298_test [idx]
y_h298_test = y_h298_test [idx]
y_g298_test = y_g298_test [idx]

# save subsampling data

with open ('features_subsample_train.pickle','wb') as f:
    pickle.dump ((X_train, y_cv_train, y_E_homo_train, y_E_lumo_train, y_gap_train, y_r2_train, y_zpve_train, y_u0_train, y_u298_train, y_h298_train, y_g298_train), f)

with open ('features_subsample_test.pickle','wb') as f:
    pickle.dump ((X_test, y_cv_test,y_E_homo_test, y_E_lumo_test, y_gap_test, y_r2_test, y_zpve_test, y_u0_test, y_u298_test, y_h298_test, y_g298_test), f)



#if you have saved subsampling.pickle

with open ('features_subsample_train.pickle','rb') as f:
    X_train, y_cv_train, y_E_homo_train, y_E_lumo_train, y_gap_train, y_r2_train, y_zpve_train, y_u0_train, y_u298_train, y_h298_train, y_g298_train = pickle.load(f)

with open ('features_subsample_test.pickle','rb') as f:
    X_test, y_cv_test, y_E_homo_test, y_E_lumo_test, y_gap_test, y_r2_test, y_zpve_test, y_u0_test, y_u298_test, y_h298_test, y_g298_test = pickle.load(f)

# see train shape, overal 133K samples,
# all samples, train =105K , test =33K
# with subsampling, train= 25.1K, test = 8.4K
print ("Features train array shape: ",X_train.shape)
print ("Features test array shape: ",X_test.shape)
print ("Heat_capacity train shape: ",y_gap_train.shape)
print ("Heat_capacity test shape:",y_gap_test.shape)

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def mAPE(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true))
"""
# Standardize the data both train and test data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Normalize y_train, y_test
s_min, s_max = np.min(y_train), np.max(y_train)
y_train = (y_train - s_min) / (s_max - s_min)
y_test = (y_test - s_min) / (s_max - s_min)
"""
#properties 
#Default parameters
y_train = y_zpve_train
y_test = y_zpve_test

"""
egr = RandomForestRegressor(bootstrap=True, max_samples = 1000, max_depth=2000, 
                             random_state=0,min_samples_split = 100, min_samples_leaf = 10,
                             max_features = 2, n_jobs = -1, verbose = 1) 
"""
#best model
#regr = RandomForestRegressor(max_depth=5000, random_state=0)
#max_depths = [3500,4100,4200,4300,4500,4800,5000]
max_depth = 5000
#max_depth = random.choice(max_depths)
regr = RandomForestRegressor(n_jobs = -1,max_depth = max_depth, 
                             random_state=0,verbose = 1,
                             warm_start = False             
)

regr.fit(X_train,y_train)
print (regr.get_params())
y_pred = regr.predict(X_test)
print ("feature_importances: ", regr.feature_importances_)
features = list(data.columns.values)

features.pop(-1)
features.pop(-1)
features.pop(-1)
features.pop(-1)
features.pop(-1)
features.pop(-1)
features.pop(-1)
features.pop(-1)
features.pop(-1)
features.pop(-1)

print (features)


df = pd.DataFrame(regr.feature_importances_,features)
df.plot(kind='barh')
plt.savefig('features.png')
print (df)
print(y_pred)

fig, ax = plt.subplots()
plt.figure(figsize=(15,6))
plt.barh(features, regr.feature_importances_)

plt.yticks(rotation=30, ha = 'right')
plt.title("zpve_feat_import")
plt.savefig("g2zpveeat_import.png")




y_pred = regr.predict(X_test)
y_pred_train = regr.predict(X_train)

#print the results
print ("test r-square from the model: ",regr.score ( X_test, y_test))
print ("test r-square from the model: ",r2_score ( y_test, y_pred))
print ("train r-square from the model: ",regr.score (X_train, y_train))
print ("train r-square from the model: ",r2_score (y_train, y_pred_train))

print ("Mean of Cv_test: ",np.mean(y_test))
print ("Mean Squared Error: ",mean_squared_error(y_test, y_pred))
print ("Mean Squared Error train: ",mean_squared_error(y_train, y_pred_train))
print ("MSE/Mean_Cv_test: ", mean_squared_error(y_test, y_pred)/np.mean(y_test))
print ("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
print ("Mean_absolute_error/Mean_Cv_test: ",mean_absolute_error(y_test, y_pred)/np.mean(y_test))
print ("MeanAbsolutePercentage error (MAPE): ",MAPE(y_test, y_pred))
print ("MedianAbsolutePercentage error (MdAPE): ",mAPE(y_test, y_pred))

end = time.time()
print ("how long it takes: ",end-start)

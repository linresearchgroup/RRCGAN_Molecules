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

start = time.time()

"""
#reading data, NO NEED IF  have saved data, just read *.pickle
data = pd.read_csv('qm9_feature_data.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data.iloc[:,:-1].values,
    data.iloc[:,-1].values
)
print (data.head())
imax_depth = random.choice(max_depths)
with open ('features_train.pickle','wb') as f:
    pickle.dump ((X_train, y_train), f)
with open ('features_test.pickle','wb') as f:
    pickle.dump ((X_test, y_test), f)
"""
"""
#if you have saved *.pickle, and want all samples
with open ('features_train.pickle','rb') as f:
    X_train, y_train = pickle.load(f)
with open ('features_test.pickle','rb') as f:
    X_test, y_test = pickle.load(f)

#subsampling, run if you want subsampling, NO NEED IF have saved subsampling.pickle
idx = np.random.choice(len(y_train) , int(len(y_train)*0.25), replace = False)
X_train = X_train [idx]
y_train = y_train [idx]
idx = np.random.choice(len(y_test) , int(len(y_test)*0.25), replace = False)
X_test = X_test [idx]
y_test = y_test [idx]
# save subsampling data
with open ('features_subsample_train.pickle','wb') as f:
    pickle.dump ((X_train, y_train), f)
with open ('features_subsample_test.pickle','wb') as f:
    pickle.dump ((X_test, y_test), f)
"""

#if you have saved subsampling.pickle
with open ('features_subsample_train.pickle','rb') as f:
    X_train, y_train = pickle.load(f)
with open ('features_subsample_test.pickle','rb') as f:
    X_test, y_test = pickle.load(f)

# see train shape, overal 133K samples, 
# all samples, train =105K , test =33K 
# with subsampling, train= 25.1K, test = 8.4K
print ("Features train array shape: ",X_train.shape)
print ("Features test array shape: ",X_test.shape)
print ("Heat_capacity train shape: ",y_train.shape)
print ("Heat_capacity test shape:",y_test.shape)

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
#Default parameters
#MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
#best model, r-squared 93.3 %
#regr = MLPRegressor(n_iter_no_change = 5000, learning_rate = 'adaptive', tol = 1e-8, max_iter=100000, verbose = 1).fit(X_train,y_train)
regr = MLPRegressor(n_iter_no_change = 5000, 
                    learning_rate = 'adaptive', 
                    solver = 'adam', 
                    tol = 1e-9, 
                    max_iter=500, 
                    verbose = 1).fit(X_train,y_train)

y_pred = regr.predict(X_test)


#print the results
print ("r-square from the model: ",regr.score (X_test, y_test))
print ("Mean of Cv_test: ",np.mean(y_test))
print ("Mean Squared Error: ",mean_squared_error(y_test, y_pred))
print ("MSE/Mean_Cv_test: ", mean_squared_error(y_test, y_pred)/np.mean(y_test))
print ("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
print ("Mean_absolute_error/Mean_Cv_test: ",mean_absolute_error(y_test, y_pred)/np.mean(y_test)) 
print ("MeanAbsolutePercentage error (MAPE): ",MAPE(y_test, y_pred))
print ("MedianAbsolutePercentage error (MdAPE): ",mAPE(y_test, y_pred))

end = time.time()
print ("how long it takes: ",end-start)


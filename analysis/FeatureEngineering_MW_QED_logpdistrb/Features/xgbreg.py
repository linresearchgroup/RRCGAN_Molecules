import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import pickle

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import time

start = time.time()
"""
data = pd.read_csv('qm9_feature_data.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data.iloc[:,:-1].values,
    data.iloc[:,-1].values
)
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


# kfolds = KFold(shuffle = True, random_state = 42)
scorer = make_scorer(mean_squared_error, greater_is_better = False)

xgb = XGBRegressor(objective = 'reg:squarederror')
param_grid = {'max_depth': range(3,11),
              'learning_rate': [1e-4, 0.001, 0.01, 0.1, 1],
              'n_estimators': range(500, 5000),
              'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]}
xgb = RandomizedSearchCV(xgb, param_grid, n_iter = 100,
                         cv = 3, scoring = scorer, refit = True,
                         n_jobs = 2, verbose = 2)
xgb.fit(X_train, y_train)
best_params = xgb.best_params_

xgb = XGBRegressor(objective = 'reg:squarederror',
                   n_jobs = 2)
xgb.set_params(**best_params)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
y_pred_train = xgb.predict(X_train)
plt.scatter(y_test, y_pred)

plt.savefig("xgboost.png")
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def mAPE(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true))

print ("r2_score: ",r2_score(y_test, y_pred))

print ("Mean of Cv_test: ",np.mean(y_test))

print ("Mean Squared Error: ",mean_squared_error(y_test, y_pred))
print ("Mean Squared Error train: ",mean_squared_error(y_train, y_pred_train))

print ("test r-square from the model: ",xgb.score ( X_test, y_test))
print ("test r-square from the model: ",r2_score ( y_test, y_pred))
print ("train r-square from the model: ",xgb.score (X_train, y_train))
print ("train r-square from the model: ",r2_score (y_train, y_pred_train))
print ("MSE/Mean_Cv_test: ", mean_squared_error(y_test, y_pred)/np.mean(y_test))

print ("Mean absolute error: ",mean_absolute_error(y_test, y_pred))
print ("Mean_absolute_error/Mean_Cv_test: ",mean_absolute_error(y_test, y_pred)/np.mean(y_test)) 



print ("MeanAbsolutePercentage error (MAPE): ",MAPE(y_test, y_pred))

print ("MedianAbsolutePercentage error (MdAPE): ",mAPE(y_test, y_pred))
end = time.time()
print ("how long it takes: ",end-start)


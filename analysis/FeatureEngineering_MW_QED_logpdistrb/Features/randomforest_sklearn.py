import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines

#import graphlab as gl
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import time
import random


start = time.time()

# insert your file name here!!!
csv_filename = 'Features_hc_qm9.csv'
data = pd.read_csv(csv_filename)
"""
X_train, X_test, y_train, y_test = train_test_split(
    data.iloc[:,:-1].values,
    data.iloc[:,-1].values
)
"""
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Logp', 'QED', 'TPSA', 'heat_capacity'], axis=1),
    data.iloc[:,-1].values
)

print (data.head())
"""
egr = RandomForestRegressor(bootstrap=True, max_samples = 1000, max_depth=2000, 
                             random_state=0,min_samples_split = 100, min_samples_leaf = 10,
                             max_features = 2, n_jobs = -1, verbose = 1) 
"""
#best model
#regr = RandomForestRegressor(max_depth=5000, random_state=0)
#max_depths = [3500,4100,4200,4300,4500,4800,5000]
max_depth = 5000
def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def mAPE(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true))

start = time.time()
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
print (features)

# qm9 file has one more column, should be removed
try:
    df = pd.DataFrame(regr.feature_importances_,features)
except:
    features.pop(-1)
    df = pd.DataFrame(regr.feature_importances_,features)
df.plot(kind='barh')
#plt.savefig('features{}.png'.format(csv_filename), dpi=700)
print (df)
print(y_pred)

fig, ax = plt.subplots()
fig.subplots_adjust(left = 0.5)
plt.figure(figsize=(25,10))
plt.barh(features, regr.feature_importances_, color='blue')
#ax.tick_params(axis='x', which='major', labelsize=10)
mpl.rcParams['axes.linewidth'] = 5

plt.xlabel('RF Importance value', fontsize=20)
#plt.ylabel('Features', fontsize=20)

plt.yticks(rotation=30, ha = 'right', fontsize=16)
plt.xticks(fontsize=20)
#fig.subplots_adjust(left = 0.5)

plt.savefig('last_features{}.png'.format(csv_filename), dpi=700)

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

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
regr = RandomForestRegressor(bootstrap=True, max_samples = 1000, max_depth=2000, 
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


features2 = ['Mol.Wt.', 'HeavyAtomCount', 'HeavyAtomMol.Wt.', 'Num.H-Acceptors', 'Num.H-Donors', 'Num.Het.atoms', 'Num.Rot.Bonds', 'NumVal.Elec.', 'Num.Arom.Rings', 'Num.Sat.Rings', 'Num.Aliph.Rings', 'Num.Rad.Elect.', 'Num.Aliph.Carbocyc.', 'Num.Aliph.Het.cyc.', 'Num.Arom.Carbocyc.', 'Num.Arom.Het.cyc.', 'Num.Sat.Carbocyc.', 'Num.Sat.Het.cyc.']

XX = pd.Series(regr.feature_importances_, index=features2)

print("here is your feature importances: index ", XX.index)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                               figsize=(10,6))

mpl.rcParams['axes.linewidth'] = 2.5
XX2 = np.around(XX.values, decimals=5)

ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x',which='both',bottom=False)
ax2.spines['top'].set_visible(False)

plt.rcParams["figure.autolayout"] = True

bs = 0.08
ts = 0.64

ax2.set_ylim(0,bs)
ax1.set_ylim(ts, 0.66)

bars1 = ax1.bar(XX.index, XX2, color='blue')
plt.sca(ax1)
plt.yticks(rotation=0, fontsize=22)
plt.yticks(np.linspace(ts, 0.66, 3))
ax1.xaxis.set_tick_params(width=3)
ax1.yaxis.set_tick_params(width=3)
#plt.legend(["QM9"], rotation=90,fontsize=15)

bars2 = ax2.bar(XX.index, XX2, color='blue')
plt.sca(ax2)
plt.xticks(rotation=90, fontsize=17)

plt.yticks(rotation=0, fontsize=22) 
plt.yticks(np.linspace(0, bs, 3))
ax2.xaxis.set_tick_params(width=3)
ax2.yaxis.set_tick_params(width=3)
ax2.set_xlabel(fontweight='bold')
"""
for tick in ax2.get_xticklabels():
    tick.set_rotation(-45, fontsize=15)
"""


d = .008  
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)      
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)  
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

for b1, b2 in zip(bars1, bars2):
    posx = b2.get_x() + b2.get_width()/2.
    if b2.get_height() > bs:
        ax2.plot((posx-3*d, posx+3*d), (1 - d, 1 + d), color='k', clip_on=False,
                 transform=ax2.get_xaxis_transform())
    if b1.get_height() > ts:
        ax1.plot((posx-3*d, posx+3*d), (- d, + d), color='k', clip_on=False,
                 transform=ax1.get_xaxis_transform())

plt.savefig('last_features{}.png'.format(csv_filename), bbox_inches='tight', dpi=700)


fig, (ax1, ax2) = plt.subplots(1,2, sharey=True,
                               figsize=(5,6))

fig.subplots_adjust(left = 0.5)
plt.figure(figsize=(25,10))
plt.barh(features, regr.feature_importances_, color='red')
#ax.tick_params(axis='x', which='major', labelsize=10)
mpl.rcParams['axes.linewidth'] = 3

plt.xlabel('RF Importance value', fontsize=15)
#plt.ylabel('Features', fontsize=20)

plt.yticks(rotation=30, ha = 'right', fontsize=10)
plt.xticks(fontsize=10)
#fig.subplots_adjust(left = 0.5)


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


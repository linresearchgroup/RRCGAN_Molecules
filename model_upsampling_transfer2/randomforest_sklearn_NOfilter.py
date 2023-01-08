import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
#import graphlab as gl
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import time
import random
start = time.time()

data = pd.read_csv('qm9_feature_data.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data.iloc[:,:-1].values,
    data.iloc[:,-1].values
)

data_gen = pd.read_csv('final_data_Feat_Cv_smiles.csv')
X_test_gen, y_test_gen, gen_smiles = (
    data_gen.iloc[:,0:-2].values,
    data_gen.iloc[:,-2].values,
    data_gen.iloc[:,-1].values
)

print (data.head())
print (data_gen.head())

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
regr = RandomForestRegressor(n_jobs = -1,max_depth = 5000, 
                             random_state=0,verbose = 1,
                             warm_start = False             
)

regr.fit(X_train,y_train)
print (regr.get_params())
y_pred = regr.predict(X_test_gen)
print ("feature_importances: ", regr.feature_importances_)
features = list(data.columns.values)

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
plt.savefig("last.png")

print ("this is y_test_gen", y_test_gen)
print ("this is pred_y with y_test_gen", y_pred)
"""
acc_ids = []
for (i,y_gen) in enumerate (y_test_gen):
    if ((y_gen-y_pred[i])/y_gen)<0.05:
        acc_ids.append(i)

print (acc_ids)
y_test_gen = y_test_gen[acc_ids]
print ("shape of Cv test gen after selecting accurate one", y_test_gen.shape)
y_pred = y_pred [acc_ids]   
print ("predicted y_pred",y_pred)
X_test_gen = X_test_gen [acc_ids]
gen_smiles = gen_smiles [acc_ids]

final_data_Feat_Cv_smiles = pd.read_csv('Features_gen_smiles.csv')
final_data_Feat_Cv_smiles = final_data_Feat_Cv_smiles.iloc[acc_ids]
final_data_Feat_Cv_smiles.to_csv('final_data_Feat_Cv_smiles.csv')
""" 
print ("r-square from the model: ",regr.score (X_test_gen, y_test_gen))
print ("Mean of Cv_test: ",np.mean(y_test_gen))
print ("Mean Squared Error: ",mean_squared_error(y_test_gen, y_pred))
print ("MSE/Mean_Cv_test: ", mean_squared_error(y_test_gen, y_pred)/np.mean(y_test_gen))
print ("Mean absolute error: ",mean_absolute_error(y_test_gen, y_pred))
print ("Mean_absolute_error/Mean_Cv_test: ",mean_absolute_error(y_test_gen, y_pred)/np.mean(y_test_gen)) 
print ("MeanAbsolutePercentage error (MAPE): ",MAPE(y_test_gen, y_pred))
print ("MedianAbsolutePercentage error (MdAPE): ",mAPE(y_test_gen, y_pred))
end = time.time()
print ("how long it takes: ",end-start)

import numpy as np
from numpy import concatenate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Input, Dropout, LSTM, Reshape, LeakyReLU,
                          Concatenate, ReLU, Flatten, Dense, Embedding,
                          BatchNormalization, Activation, SpatialDropout1D,
                          Conv2D, MaxPooling2D, UpSampling2D)
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
import tensorflow.keras.backend as K
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('qm9_feature_data.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data.iloc[:,:-1].values,
    data.iloc[:,-1].values
)
print (X_train.shape)
VALELEC = X_train[:,7]
print (VALELEC.shape)
V=np.reshape(VALELEC,(-1,1))
print (V.shape)
"""
print (data.iloc[:,2].values)
print (data['HeavyAtomMolWt'])
print (data.iloc[:,10].values)
print (data['NumAliphaticRings'])
print (data.iloc[:,13].values)
print (data['NumAliphaticHeterocycles'])
print (data.iloc[:,15].values)
print (data['NumAromaticHeterocycles'])
"""
NumValElec_train, NumValElec_test = train_test_split(data.iloc[:,7].values)
NumValElec_train=np.reshape(NumValElec_train,(-1,1))
NumValElec_test=np.reshape(NumValElec_test,(-1,1))
HeavyAtomMolWt_train, HeavyAtomMolWt_test = train_test_split (data.iloc[:,2].values)
HeavyAtomMolWt_train=np.reshape(HeavyAtomMolWt_train,(-1,1))
HeavyAtomMolWt_test=np.reshape(HeavyAtomMolWt_test,(-1,1))

NumAliRings_train, NumAliRings_test = train_test_split(data.iloc[:,10].values)
NumAliRings_train=np.reshape(NumAliRings_train,(-1,1))
NumAliRings_test=np.reshape(NumAliRings_test,(-1,1))
NumAliHetCycl_train, NumAliHetCycl_test = train_test_split(data.iloc[:,13].values)
NumAliHetCycl_train=np.reshape(NumAliHetCycl_train,(-1,1))
NumAliHetCycl_test=np.reshape(NumAliHetCycl_test,(-1,1))

NumAroHetCycl_train, NumAroHetCycl_test = train_test_split(data.iloc[:,15].values)
NumAroHetCycl_train = np.reshape(NumAroHetCycl_train,(-1,1))
NumAroHetCycl_test = np.reshape(NumAroHetCycl_test,(-1,1))
print (NumValElec_train.shape)
print (NumValElec_train)
#yr = Concatenate()([inp1, inp2])

NumValElec_HeavyAtomMolWt_train = np.concatenate((NumValElec_train, HeavyAtomMolWt_train),axis=1)
NumValElec_HeavyAtomMolWt_test = np.concatenate((NumValElec_test, HeavyAtomMolWt_test),axis=1)
Rings_Cycles_train = np.concatenate((NumAliRings_train, NumAliHetCycl_train, NumAroHetCycl_train),axis=1)
Rings_Cycles_test = np.concatenate((NumAliRings_test, NumAliHetCycl_test, NumAroHetCycl_test),axis=1)

all_input_train = np.concatenate((NumValElec_HeavyAtomMolWt_train, Rings_Cycles_train),axis=1)
all_input_test = np.concatenate((NumValElec_HeavyAtomMolWt_test, Rings_Cycles_test),axis=1)




print (NumValElec_HeavyAtomMolWt_train.shape)
"""
test_idx = np.random.choice(len(CVs), int(len(CVs) * 0.2), replace = False)
train_idx = np.setdiff1d(np.arange(len(CVs)), test_idx)

train_charges, train_coords, train_cvs = padded_charges[train_idx], padded_coords[train_idx], CVs[train_idx]
test_charges, test_coords, test_cvs = padded_charges[test_idx], padded_coords[test_idx], CVs[test_idx]
"""

# In[18]:
# create model
model = Sequential()
model.add(Dense(20, activation='tanh', input_dim = 1, kernel_initializer = 'uniform'))
model.add(Dense(1, activation='linear', kernel_initializer = 'uniform'))

# compile model
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit (NumValElec_train, y_train, epochs = 5, batch_size = 64, verbose=1)



print (model.summary())


# In[31]:

"""
model.fit(
    [
       NumValElec_HeavyAtomMolWt_train ,
       Rings_Cycles_train
    ],
    y_train,
    validation_data = (
        [
            NumValElec_HeavyAtomMolWt_test,
            Rings_Cycles_test
        ],
        y_test
    ),
    batch_size = 64,
    epochs = 2,
    verbose = 1
)
"""

# In[32]:


model.evaluate(
    [
        NumValElec_test
    ],
    y_test
)


# In[33]:


y_pred = model.predict(
    [
       NumValElec_test
    ]
)


# In[34]:


def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


# In[35]:


MAPE(y_test, y_pred.reshape([-1]))


# In[36]:


from sklearn.metrics import r2_score


# In[37]:


print (r2_score(y_test, y_pred.reshape([-1])))


# In[ ]:





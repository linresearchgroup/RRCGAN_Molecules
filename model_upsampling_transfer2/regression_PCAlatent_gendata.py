# used the encoder and decoder trained on qm9 library
# 1) get the heat capacity from encoder + regression trained on qm9
# 2) PCA analysis on qm9 data vs. generated data for latent varialbe from encoder + regression_top

import warnings
warnings.filterwarnings('ignore')

import os
import re
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import pickle
from tensorflow.keras.layers import (Input, Dropout, LSTM, Reshape, LeakyReLU,
                          Concatenate, ReLU, Flatten, Dense, Embedding,
                          BatchNormalization, Activation, SpatialDropout1D,
                          Conv2D, MaxPooling2D, UpSampling2D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import np_utils

from tensorflow.keras.utils import  to_categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt

from progressbar import ProgressBar
import seaborn as sns

from sklearn.metrics import r2_score

print ("!!!!!!!!! we are just before importing rdkit!!!!!")
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
print ("!!!!!!!!!!!!!!!!!!!!!we are after importing rdkit!!!!!!!!!!!!!!!!!!")


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3667)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)


# loading data both qm9 and generated data
with open('./../data/trainingsets/20000_train_regular_qm9/image_train.pickle', 'rb') as f:
    X_smiles_train, X_atoms_train, X_bonds_train, y_train = pickle.load(f)

with open('./../data/trainingsets/20000_train_regular_qm9/image_test.pickle', 'rb') as f:
    X_smiles_test, X_atoms_test, X_bonds_test, y_test = pickle.load(f)
    
with open('gen_smiles2_atombond.pickle', 'rb') as f:
    X_smiles_val, X_atoms_val, X_bonds_val, y_val_, y_dft_ = pickle.load(f)

with open('./../data/trainingsets/20000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
    
tokenizer[0] = ' '

# Subsampling has been done in the data preprocesses for qm9 data

def norm(X: ndarray) -> ndarray:
    X = np.where(X == 0, -1.0, 1.0)
    return X

X_atoms_train, X_bonds_train = (norm(X_atoms_train),
                                norm(X_bonds_train))
X_atoms_test, X_bonds_test = (norm(X_atoms_test),
                                norm(X_bonds_test))
X_atoms_val, X_bonds_val = (norm(X_atoms_val),
                            norm(X_bonds_val))

def y_norm(y: ndarray) -> ndarray:
    scaler_min = np.min(y)
    scaler_max = np.max(y)
    
    y = (y - scaler_min) / (scaler_max - scaler_min)
    
    return y, scaler_min, scaler_max

y_train, s_min, s_max = y_norm(y_train)
print (s_min, s_max)
y_test = (y_test - s_min) / (s_max - s_min)
y_val = (y_val_ - s_min) / (s_max - s_min)
y_dft = (y_dft_ - s_min) / (s_max - s_min)
#y_test, s_min, s_max = y_norm(y_test)
#y_val, s_min_val, s_max_val = y_norm(y_val)

encoder = load_model('./../data/nns/encoder.h5')
decoder = load_model('./../data/nns/decoder.h5')


# Regressor
inp1 = Input(shape = [6, 6, 1])
inp2 = Input(shape = [6, 6, 1])

yr = Concatenate()([inp1, inp2])

tower0 = Conv2D(32, 1, padding = 'same')(yr)
tower1 = Conv2D(64, 1, padding = 'same')(yr)
tower1 = Conv2D(64, 3, padding = 'same')(tower1)
tower2 = Conv2D(32, 1, padding = 'same')(yr)
tower2 = Conv2D(32, 5, padding = 'same')(tower2)
tower3 = MaxPooling2D(3, 1, padding = 'same')(yr)
tower3 = Conv2D(32, 1, padding = 'same')(tower3)
h = Concatenate()([tower0, tower1, tower2, tower3])
h = ReLU()(h)
h = MaxPooling2D(2, 1, padding = 'same')(h)

for i in range(6):
    tower0 = Conv2D(32, 1, padding = 'same')(h)
    tower1 = Conv2D(64, 1, padding = 'same')(h)
    tower1 = Conv2D(64, 3, padding = 'same')(tower1)
    tower2 = Conv2D(32, 1, padding = 'same')(h)
    tower2 = Conv2D(32, 5, padding = 'same')(tower2)
    tower3 = MaxPooling2D(3, 1, padding = 'same')(h)
    tower3 = Conv2D(32, 1, padding = 'same')(tower3)
    h = Concatenate()([tower0, tower1, tower2, tower3])
    h = ReLU()(h)
    if i % 2 == 0 and i != 0:
        h = MaxPooling2D(2, 1, padding = 'same')(h)
h = BatchNormalization()(h)

yr = Flatten()(h)
o = Dropout(0.2)(yr)
o = Dense(128)(o)

o_reg = Dropout(0.2)(o)
o_reg = Dense(1, activation = 'sigmoid')(o_reg)

regressor = Model([inp1, inp2], o_reg)
regressor_top = Model([inp1, inp2], o)

regressor.compile(loss = 'mse', optimizer = Adam(1e-5))


train_atoms_embedding, train_bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])

atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
atoms_test, bonds_test, _ = encoder.predict([X_atoms_test, X_bonds_test])
atoms_val, bonds_val, _ = encoder.predict([X_atoms_val, X_bonds_val])

regressor = load_model    ('./../data/nns/regressor.h5')
regressor_top = load_model('./../data/nns/regressor_top.h5')

regressor.fit([atoms_embedding, bonds_embedding], 
              y_train,
              validation_data = ([atoms_test,
                                  bonds_test],
                                 y_test),
              batch_size = 32,
              epochs = 1,
              verbose = 1)

# Validating the regressor
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current train (qm9) R2 on Regressor: {}'.format(r2_score(y_train, pred.reshape([-1]))))

atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_test, X_bonds_test])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current test (qm9) R2 on Regressor: {}'.format(r2_score(y_test, pred.reshape([-1]))))
#====#
atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_val, X_bonds_val])
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current gen_data R2 on Regressor: {}'.format(r2_score(y_dft, pred.reshape([-1]))))
print (y_train)
print (y_test)
print (y_val)
print (y_val_)
print (y_dft)
print (y_dft_)




 


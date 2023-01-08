# Strategy 1:
# Generate data after each epoch of training, if less than
# 10% error rate, and is a legit SMILES
# append to the real data
# Otherwise, append to fake data

# ADDING REINFORCEMENT MECHANISM
# Regenerate Normal sampling (define ranges), default: uniform

# IMPORTANT!!!!!!!!!!!!! DO NOT DROP DUPLICATE FOR RESULT .CSV
import warnings
warnings.filterwarnings('ignore')

import time
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
from sklearn.model_selection import KFold
import matplotlib as mpl
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

from rdkit import Chem

print ("!!!!!!!!! we are just before importing rdkit!!!!!")
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
print ("!!!!!!!!!!!!!!!!!!!!!we are after importing rdkit!!!!!!!!!!!!!!!!!!")
from scipy.stats import truncnorm
from sklearn.decomposition import PCA

import matplotlib.ticker as tk

import ntpath
import re

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

from matplotlib.colors import ListedColormap

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3667)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

with open('./../data/trainingsets/60000_train_regular_qm9/image_train.pickle', 'rb') as f:
    X_smiles_train0, X_atoms_train0, X_bonds_train0, y_train0 = pickle.load(f)
    
with open('./../data/trainingsets/60000_train_regular_qm9/image_test.pickle', 'rb') as f:
    X_smiles_val0, X_atoms_val0, X_bonds_val0, y_val0 = pickle.load(f)

with open('./../data/trainingsets/60000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
    
tokenizer[0] = ' '


s_min1 = np.min (y_train0)
s_max1 = np.max (y_train0)

s_min2 = np.min(y_val0)
s_max2 = np.max(y_val0)
s_min = min(s_min1, s_min2)
s_max = max(s_max1, s_max2)
# Subsampling has been done in the data preprocesses

kfold = KFold (5, True, 1)
print (y_train0.shape)
for train, test in kfold.split(y_train0):
		# Outlier removal
		print (train, test)
		y_train = y_train0[train]
		X_smiles_train = X_smiles_train0[train]
		X_atoms_train = X_atoms_train0[train]
		X_bonds_train = X_bonds_train0[train]

		y_val = y_train0[test]
		X_smiles_val = X_smiles_train0[test]
		X_atoms_val = X_atoms_train0[test]
		X_bonds_val = X_bonds_train0[test]
		"""
		IQR = - np.quantile(y_train, 0.25) + np.quantile(y_train, 0.75)

		lower_bound, upper_bound = np.quantile(y_train, 0.25) - 1.5 * IQR, np.quantile(y_train, 0.75) + 1.5 * IQR

		idx = np.where((y_train >= lower_bound) & (y_train <= upper_bound))

		y_train = y_train[idx]
		X_smiles_train = X_smiles_train[idx]
		X_atoms_train = X_atoms_train[idx]
		X_bonds_train = X_bonds_train[idx]


		# Outlier removal
		IQR = - np.quantile(y_val, 0.25) + np.quantile(y_val, 0.75)

		lower_bound, upper_bound = np.quantile(y_val, 0.25) - 1.5 * IQR, np.quantile(y_val, 0.75) + 1.5 * IQR

		idx = np.where((y_val >= lower_bound) & (y_val <= upper_bound))

		y_val = y_val[idx]
		X_smiles_val = X_smiles_val[idx]
		X_atoms_val = X_atoms_val[idx]
		X_bonds_val = X_bonds_val[idx]
		"""

		def norm(X: ndarray) -> ndarray:
			X = np.where(X == 0, -1.0, 1.0)
			return X

		X_atoms_train, X_bonds_train = (norm(X_atoms_train),
										norm(X_bonds_train))
		X_atoms_val, X_bonds_val = (norm(X_atoms_val),
									norm(X_bonds_val))

		def y_norm(y: ndarray) -> ndarray:
			scaler_min = np.min(y)
			scaler_max = np.max(y)
			
			y = (y - scaler_min) / (scaler_max - scaler_min)
			
			return y, scaler_min, scaler_max
		"""
		s_min1 = np.min (y_train)
		s_max1 = np.max (y_train)

		s_min2 = np.min(y_val)
		s_max2 = np.max(y_val)
		s_min = min(s_min1, s_min2)
		s_max = max(s_max1, s_max2)
		"""
		y_val = (y_val - s_min) / (s_max - s_min)
		print ("min and max train data and test normalized", s_min, s_max, np.min(y_val), np.max(y_val))
		# define s_min and s_max between 10-7s_max = 50
		#s_min, s_max = 20, 50
		y_train = (y_train - s_min) / (s_max - s_min)
		print ("min and max train data and train normalized", s_min, s_max, np.min(y_train), np.max(y_train))

		encoder = load_model('./../data/nns_60ksam/encoder.h5')
		decoder = load_model('./../data/nns_60ksam/decoder.h5')

		class Config:
			
			def __init__(self):
				self.Filters = [256, 128, 64]
				self.genFilters = [128, 128, 128]
				self.upFilters = [(2, 2), (2, 2), (2, 2)]
				
		config = Config()

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
		print (regressor.summary())

		train_atoms_embedding, train_bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])

		atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
		atoms_val, bonds_val, _ = encoder.predict([X_atoms_val, X_bonds_val])

		regressor = load_model('./../data/nns_60ksam/regressor.h5')
		regressor_top = load_model('./../data/nns_60ksam/regressor_top.h5')
		# No validation data, chose validation form training
		"""
		regressor.fit([atoms_embedding, bonds_embedding], 
					  y_train,
					  validation_data = ([atoms_val,
										  bonds_val],
										 y_val,
					  batch_size = 32,
					  epochs = 50,
					  verbose = 1)
		"""
		regressor.fit([atoms_embedding, bonds_embedding],
					  y_train,
					  validation_split = 0.2,
					  batch_size = 32,
					  epochs = 1,
					  verbose = 1)
		# Validating the regressor
		#====#
		atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_train, X_bonds_train])
		pred = regressor.predict([atoms_embedding, bonds_embedding])
		print('Current R2 on Regressor for train data: {}'.format(r2_score(y_train, pred.reshape([-1]))))
		print (pred)
		print (y_train)
		#====#
		atoms_embedding, bonds_embedding, _ = encoder.predict([X_atoms_val, X_bonds_val])
		pred = regressor.predict([atoms_embedding, bonds_embedding])
		print('Current R2 on Regressor for validation data: {}'.format(r2_score(y_val, pred.reshape([-1]))))
		print ("pred of validation data: ", pred )
		print ("True validation values: ", y_val)
		# Saving the currently trained models
		#regressor.save('./../data/nns/regressor.h5')
		#regressor_top.save('./../data/nns/regressor_top.h5')

		#regressor = load_model('./../data/nns/regressor.h5')
		#regressor_top = load_model('./../data/nns/regressor_top.h5')


X_atoms_val0, X_bonds_val0 = (norm(X_atoms_val0),
                            norm(X_bonds_val0))

y_val0 = (y_val0 - s_min) / (s_max - s_min)

atoms_embedding_test, bonds_embedding_test, _ = encoder.predict([X_atoms_val0, X_bonds_val0])
pred_test = regressor.predict([atoms_embedding_test, bonds_embedding_test])

X_atoms_train0, X_bonds_train0 = (norm(X_atoms_train0),
                            norm(X_bonds_train0))

y_train0 = (y_train0 - s_min) / (s_max - s_min)

atoms_embedding_train, bonds_embedding_train, _ = encoder.predict([X_atoms_train0, X_bonds_train0])
pred_train = regressor.predict([atoms_embedding_train, bonds_embedding_train])

print('Current R2 on Regressor for training data: {}'.format(r2_score(y_train0, pred_train.reshape([-1]))))
print ("pred of training data: ", pred_train)
print ("True training values: ", y_train0)

print('Current R2 on Regressor for validation data: {}'.format(r2_score(y_val0, pred_test.reshape([-1]))))
print ("pred of validation data: ", pred_test )
print ("True validation values: ", y_val0)

# graph the predicted value VS. true value

plt.rcParams['ps.useafm'] = True
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams['pdf.fonttype'] = 42

mpl.rcParams['axes.linewidth'] = 3
fig, ax = plt.subplots(figsize = (13, 13))

ax.tick_params(axis='both', which='major', labelsize=35, width=5)
mpl.font_manager.FontManager()

CV_range = (8, 48)
lims=[CV_range[0], CV_range[1]]

pred_test =     pred_test*(s_max-s_min)+s_min
y_val0 = y_val0*(s_max-s_min)+s_min

plt.locator_params(axis="x", nbins=6)
plt.locator_params(axis="y", nbins=6)

error = np.abs((pred_test-y_val0)/y_val0*100)
plt.scatter(pred_test, y_val0, 
            s=10, c='black')
plt.plot(lims, lims, '--',c='red', alpha=1, zorder=0)

#plt.plot(lims, lims, '--',c='red', alpha=1, zorder=0)
plt.ylabel(r'Predicted $\mathbf{C_v (\frac{cal}{mol. K})}$', fontsize=35, fontweight='bold')
plt.xlabel(r'True test $\mathbf{C_v (\frac{cal}{mol. K})}$', fontsize=35, fontweight='bold')

plt.savefig('5CV_test_regressor_pred_true_cv', dpi=1000, bbox_inches='tight')

# plot for training data
mpl.rcParams['axes.linewidth'] = 3
fig, ax = plt.subplots(figsize = (13, 13))

ax.tick_params(axis='both', which='major', labelsize=35, width=5)
mpl.font_manager.FontManager()

CV_range = (8, 48)
lims=[CV_range[0], CV_range[1]]

pred_train =     pred_train*(s_max-s_min)+s_min
y_train0 = y_train0*(s_max-s_min)+s_min

plt.locator_params(axis="x", nbins=6)
plt.locator_params(axis="y", nbins=6)

error = np.abs((pred_train-y_val0)/y_val0*100)
plt.scatter(pred_train, y_train0,
            s=10, c='red')

plt.plot(lims, lims, '--',c='black', alpha=1, zorder=0)

#plt.plot(lims, lims, '--',c='red', alpha=1, zorder=0)
plt.ylabel(r'Predicted $\mathbf{C_v (\frac{cal}{mol. K})}$', fontsize=35, fontweight='bold')
plt.xlabel(r'True train $\mathbf{C_v (\frac{cal}{mol. K})}$', fontsize=35, fontweight='bold')

plt.savefig('5CV_train_regressor_pred_true_cv', dpi=1000, bbox_inches='tight')

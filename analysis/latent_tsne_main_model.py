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
import random
from progressbar import ProgressBar
import numpy as np
from numpy import ndarray

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.manifold import TSNE
from scipy.stats import truncnorm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen as logp

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Input, Dropout, LSTM, Reshape, LeakyReLU,
                          Concatenate, ReLU, Flatten, Dense, Embedding,
                          BatchNormalization, Activation, SpatialDropout1D,
                          Conv2D, MaxPooling2D, UpSampling2D, Lambda)
from tensorflow.keras.models     import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses     import mse, binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import  mean_squared_error as mse_keras
from tensorflow.keras.backend import argmax as argmax
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import one_hot

from tensorflow.keras.utils import  to_categorical
from tensorflow import random as randomtf

from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from   matplotlib.lines import Line2D
from   matplotlib.colors import ListedColormap
import matplotlib.ticker as tk
from matplotlib import rc, rcParams

from progressbar import ProgressBar
import seaborn as sns

from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor, construct_atomic_number_array
preprocessor = GGNNPreprocessor()
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem

import ntpath
from scipy.stats import truncnorm

""" fix all the seeds,results are still slighthly different """
randomtf.set_seed(0)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(4)
random.seed(12345)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3667)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

tf.config.list_physical_devices('GPU')

""" reading and preprocessing data"""
with open('./../data/trainingsets/60000_train_regular_qm9/image_train.pickle', 'rb') as f:
    X_smiles_train, SMILES_train, X_atoms_train, X_bonds_train, y_train0 = pickle.load(f)

with open('./../data/trainingsets/60000_train_regular_qm9/image_test.pickle', 'rb') as f:
    X_smiles_val, SMILES_val, X_atoms_val, X_bonds_val, y_val0 = pickle.load(f)

with open('./../data/trainingsets/60000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
tokenizer[0] = ' '

with open('./../data/trainingsets/60000_train_regular_qm9/train_GAN.pickle', 'rb') as f:
    X_smiles_gantrain, SMILES_gantrain, cv_gantrain = pickle.load(f)


X_smiles_gantrain_ = []
for i in X_smiles_gantrain:
  X_smiles_gantrain_.append(i)
X_smiles_gantrain = np.array (X_smiles_gantrain_)
print (X_smiles_gantrain.shape)

cv_gantrain_ = []
for i in cv_gantrain:
  cv_gantrain_.append(i)
cv_gantrain = np.array (cv_gantrain_)
print (cv_gantrain.shape)

SMILES_gantrain_ = []
for smiles in SMILES_gantrain:
  SMILES_gantrain_.append(smiles)
SMILES_gantrain = np.array (SMILES_gantrain_)
print (SMILES_gantrain.shape)

# Subsampling has been done in the data preprocesses
print ('X_smiles_train shape: ', X_smiles_train.shape)
print ('X_smiles_test shape: ', X_smiles_val.shape)
#print ('last SMILES train: ', SMILES_train[-1])

## Outlier removal 2*IQR rule
# Train samples
IQR = - np.quantile(y_train0, 0.25) + np.quantile(y_train0, 0.75)
lower_bound, upper_bound = np.quantile(y_train0, 0.25) - 2 * IQR, np.quantile(y_train0, 0.75) + 2 * IQR
idx = np.where((y_train0 >= lower_bound) & (y_train0 <= upper_bound))

y_train = y_train0[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]

# Test samples
IQR = - np.quantile(y_val0, 0.25) + np.quantile(y_val0, 0.75)
lower_bound, upper_bound = np.quantile(y_val0, 0.25) - 2 * IQR, np.quantile(y_val0, 0.75) + 2 * IQR
idx = np.where((y_val0 >= lower_bound) & (y_val0 <= upper_bound))

y_val = y_val0[idx]
X_smiles_val = X_smiles_val[idx]
X_atoms_val = X_atoms_val[idx]
X_bonds_val = X_bonds_val[idx]


# subsampling
idx = np.random.choice(len(y_train), int(len(y_train) * 0.016), replace = False)
y_train = y_train[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]

idx = np.random.choice(len(cv_gantrain), int(len(cv_gantrain) * 0.016), replace=False)
cv_gantrain = cv_gantrain[idx]
X_smiles_gantrain = X_smiles_gantrain[idx]
SMILES_gantrain = SMILES_gantrain[idx]

# normalize the bond and aotm matrices:
def norm(X: ndarray) -> ndarray:
    X = np.where(X == 0, -1.0, 1.0)
    return X

X_atoms_train, X_bonds_train = (norm(X_atoms_train),
                                norm(X_bonds_train))
X_atoms_val, X_bonds_val = (norm(X_atoms_val),
                            norm(X_bonds_val))
# normalize the property
s_min1 = np.min (y_train)
s_max1 = np.max (y_train)

s_min2 = np.min(y_val)
s_max2 = np.max(y_val)

s_min = min(s_min1, s_min2)
s_max = max(s_max1, s_max2)

s_min_dataset, s_max_dataset = s_min, s_max
s_min_norm, s_max_norm = s_min_dataset, s_max_dataset

y_val   = (y_val -   s_min_norm) / (s_max_norm - s_min_norm)
y_train = (y_train - s_min_norm) / (s_max_norm - s_min_norm)
cv_gantrain = (cv_gantrain - s_min_norm) / (s_max_norm - s_min_norm)
print ("min and max dataset and val normalized", s_min, s_max, np.min(y_val), np.max(y_val))
print ("min and max dataset and train normalized", s_min, s_max, np.min(y_train), np.max(y_train))
print ("min and max used for normalization: ", s_min_norm, s_max_norm)

encoder = load_model('./../data/nns_9HA_noemb_6b6/keep/encoder_newencinp.h5')
decoder = load_model('./../data/nns_9HA_noemb_6b6/keep/decoder_newencinp.h5')
regressor =     load_model('./../data/nns_9HA_noemb_6b6/keep/regressor.h5')
regressor_top = load_model('./../data/nns_9HA_noemb_6b6/keep/regressor_top.h5')
generator = load_model    ('./../data/nns_9HA_noemb_6b6/keep/generator_new.h5')
discriminator= load_model ('./../data/nns_9HA_noemb_6b6/keep/discriminator_new.h5')

# No need following block if gen. samples are OK
"""
N = 30
n_sample = 700

gen_error = []
gen_smiles = []
sample_ys = []
preds = []
gen_atoms_embedding = []
gen_bonds_embedding = []

regressor_top.trainable = False
regressor.trainable = False
generator.trainable = False
discriminator.trainable = False

np.random.seed(10)

pbar = ProgressBar()
for hc in pbar(range(n_sample)):
    try:
        # get it back to original of s_min to s_max
        sample_y = np.random.uniform(s_min_norm, s_max_norm, size=[1,])
        print (sample_y)
        sample_y = np.round(sample_y, 4)
        sample_y = sample_y * np.ones([N,])
        sample_y_ = (sample_y - s_min_norm) / (s_max_norm - s_min_norm)
        sample_z = np.random.normal(0, 1, size = (N, 128))

        regressor_top.trainable = False
        regressor.trainable = False

        sample_atoms_embedding, sample_bonds_embedding = generator.predict([sample_z, sample_y_])
        dec_embedding = np.concatenate([sample_atoms_embedding, sample_bonds_embedding], axis = -1)

        softmax_smiles = decoder.predict(dec_embedding)[0]
        argmax_smiles = np.argmax(softmax_smiles, axis = 2)
        print ('shape argmax_smiles', argmax_smiles.shape)
        smiles = to_categorical(argmax_smiles, num_classes=23)
        SHAPE = list(smiles.shape) + [1]
        print ('shape line 767', SHAPE)
        smiles = smiles.reshape(SHAPE)

        latent_encoder_atom, latent_encoder_bond, _ = encoder.predict([smiles])
        pred = regressor.predict([latent_encoder_atom, latent_encoder_bond]).reshape([-1])
        pred = pred * (s_max_norm - s_min_norm) + s_min_norm

        gen_errors = np.abs((pred - sample_y) / sample_y).reshape([-1])

        #accurate = np.where(gen_errors <= 0.2)[0]
        #gen_errors = gen_errors[accurate]
        #pred = pred[accurate]

        #sample_y = sample_y[accurate]
        #sample_atoms_embedding = sample_atoms_embedding[accurate]
        #sample_bonds_embedding = sample_bonds_embedding[accurate]

        smiles = decoder.predict(dec_embedding)[0]
        smiles = np.argmax(smiles, axis = 2).reshape(smiles.shape[0], 35)

        generated_smiles = []
        for S in smiles:
            c_smiles = ''
            for s in S:
                c_smiles += tokenizer[s]
            c_smiles = c_smiles.rstrip()
            generated_smiles.append(c_smiles)
        generated_smiles = np.array(generated_smiles)
        #generated_smiles = generated_smiles [accurate]
        all_gen_smiles = []
        idx = []
        for i, smiles in enumerate(generated_smiles):
            all_gen_smiles.append(smiles[:-1])

            if ' ' in smiles[:-1]:
                continue
            #m = Chem.MolFromSmiles(smiles[:-1], sanitize=False)
            m = Chem.MolFromSmiles(smiles[:-1], sanitize=True)
            if m is not None:
                if len(construct_atomic_number_array(m)) <= 9:
                    idx.append(i)

        idx = np.array(idx)
        all_gen_smiles = np.array(all_gen_smiles)
        print ('all gen smiels shape', all_gen_smiles.shape)
        print ('gen_errors shape', gen_errors.shape)
        gen_smiles.extend(list(all_gen_smiles[idx]))
        gen_error.extend(list(gen_errors[idx]))
        sample_ys.extend(list(sample_y[idx]))
        gen_atoms_embedding.extend(sample_atoms_embedding[idx])
        gen_bonds_embedding.extend(sample_bonds_embedding[idx])
        

        preds.extend(list(pred[idx]))
    except:
        print('Did not discover SMILES for HC: {}'.format(sample_y))
        pass
 
output = {}

for i, s in enumerate (gen_smiles):

    ss = Chem.MolToSmiles(Chem.MolFromSmiles(s))
    gen_smiles[i] = ss

output['SMILES'] = gen_smiles
output['des_cv'] = sample_ys
output['pred_cv'] = preds
output['Err_pred_des'] = gen_error

with open('./../experiments/regular_9HA_6b6latent/latent/gen_atoms_bonds2.pickle', 'wb') as f:
    pickle.dump((gen_atoms_embedding, gen_bonds_embedding), f)

output = pd.DataFrame(output)
output.reset_index(drop = True, inplace = True)
output.to_csv ('./../experiments/regular_9HA_6b6latent/latent/Regular_noscreenrelug2.csv', index=False)
"""

# read the generated data
csv_name = './../experiments/regular_9HA_6b6latent/latent/Regular_noscreenrelug.csv'
gen_SMILES = pd.read_csv(csv_name)
gen_smiles = gen_SMILES ['SMILES']
sample_ys = gen_SMILES ['des_cv']
preds = gen_SMILES ['pred_cv']
gen_error = gen_SMILES ['Err_pred_des']

with open('./../experiments/regular_9HA_6b6latent/latent/gen_atoms_bonds.pickle', 'rb') as f:
    gen_atoms_embedding, gen_bonds_embedding = pickle.load(f)

print ('preds', preds)
print ('des cv', sample_ys)

plt.close()
plt.hist(gen_error)
plt.savefig("gen_error_hist.png")

# total # of samples
N = len(gen_error)
# Explained Variance R2 from sklearn.metrics.explained_variance_score
explained_variance_R2_DFT_des = explained_variance_score(sample_ys, preds)
print ("explained_varice_R2_DFT_des", explained_variance_R2_DFT_des)
rsquared = r2_score (sample_ys, preds)
print (rsquared)

gen_atoms_embedding = np.array(gen_atoms_embedding)
gen_bonds_embedding = np.array(gen_bonds_embedding)

# create classes for heat capacity
# using cv_gantrain, uniformly distributed.
y_train = cv_gantrain * (s_max_norm - s_min_norm) + s_min_norm
plt.clf()
plt.hist(y_train)
plt.savefig('cv_gantrainhist.png', dpi=100)

plt.clf()
plt.hist(preds, color='blue', alpha=1)
plt.hist(sample_ys, color='red', alpha=1)
plt.savefig('pred_des.png', dpi=100)

y_class = y_train
print (y_class) 
"""
# 10 classes
y_class = np.where(y_train <= Qs[0], 0, y_class)
y_class = np.where((y_train > Qs[0]) & (y_train <= Qs[1]), 1, y_class)
y_class = np.where((y_train > Qs[1]) & (y_train <= Qs[2]), 2, y_class)
y_class = np.where((y_train > Qs[2]) & (y_train <= Qs[3]), 3, y_class)
y_class = np.where((y_train > Qs[3]) & (y_train <= Qs[4]), 4, y_class)
y_class = np.where((y_train > Qs[4]) & (y_train <= Qs[5]), 5, y_class)
y_class = np.where((y_train > Qs[5]) & (y_train <= Qs[6]), 6, y_class)
y_class = np.where((y_train > Qs[6]) & (y_train <= Qs[7]), 7, y_class)
y_class = np.where((y_train > Qs[7]) & (y_train <= Qs[8]), 8, y_class)
y_class = np.where(y_train > Qs[8], 9, y_class)
"""
"""
# 5 classes
y_class = np.where(y_train <= Qs[1], 0, y_class)
y_class = np.where((y_train > Qs[1]) & (y_train <= Qs[3]), 1, y_class)
y_class = np.where((y_train > Qs[3]) & (y_train <= Qs[5]), 2, y_class)
y_class = np.where((y_train > Qs[5]) & (y_train <= Qs[7]), 3, y_class)
y_class = np.where(y_train > Qs[7], 4, y_class)
"""
# use the same classes
Qs_gen = np.quantile(preds, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Qs = np.quantile(y_train, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#Qs_gen = Qs
#Qs = Qs_gen
y_class_val = preds
y_class = y_train
print ("quantile of train samples: ", Qs)

# 4 classes: same # samples
y_class_val = np.where(preds <= (Qs_gen[1]+Qs_gen[2])/2, 0, y_class_val)
y_class_val = np.where((preds > (Qs_gen[1]+Qs_gen[2])/2) & (preds <= Qs_gen[4]), 1, y_class_val)
y_class_val = np.where((preds > Qs_gen[4]) & (preds <= (Qs_gen[6]+Qs_gen[7])/2), 2, y_class_val)
y_class_val = np.where(preds > (Qs_gen[6]+Qs_gen[7])/2, 3, y_class_val)

# 4 classes
y_class = np.where(y_train <= (Qs[1]+Qs[2])/2, 0, y_class)
y_class = np.where((y_train > (Qs[1]+Qs[2])/2) & (y_train <= Qs[4]), 1, y_class)
y_class = np.where((y_train > Qs[4]) & (y_train <= (Qs[6]+Qs[7])/2), 2, y_class)
y_class = np.where(y_train > (Qs[6]+Qs[7])/2, 3, y_class)

print ('gen class==0', sum(y_class_val==0))
print ('gen class==1', sum(y_class_val==1))
print ('gen class==2', sum(y_class_val==2))
print ('gen class==3', sum(y_class_val==3))

print ('train class==0', sum(y_class==0))
print ('train class==1', sum(y_class==1))
print ('train class==2', sum(y_class==2))
print ('train class==3', sum(y_class==3))

# ANALYSIS
train_atoms_embedding, train_bonds_embedding, _ = encoder.predict([X_smiles_gantrain]) 

X_atoms_train_ = train_atoms_embedding.reshape([train_atoms_embedding.shape[0], 
                                        6 * 6])
X_bonds_train_ = train_bonds_embedding.reshape([train_bonds_embedding.shape[0], 
                                        6 * 6])

X_atoms_test_ = gen_atoms_embedding.reshape([gen_atoms_embedding.shape[0],
                                      6 * 6])
X_bonds_test_ = gen_bonds_embedding.reshape([gen_bonds_embedding.shape[0], 
                                      6 * 6])

"""
### PCA ###
pca_1 = PCA(n_components = 2)
X_atoms_train = pca_1.fit_transform(X_atoms_train_)
X_atoms_test = pca_1.transform(X_atoms_test_)

pca_2 = PCA(n_components = 2)
X_bonds_train = pca_2.fit_transform(X_bonds_train_)
X_bonds_test = pca_2.transform(X_bonds_test_)

# PCA1 vs. PCA2 Atoms gen and train
plt.close()
fig, ax = plt.subplots(figsize =(8, 6))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
plt.scatter(X_atoms_train[:,0], X_atoms_train[:,1], alpha = 0.2, c = 'blue')
ax.tick_params(width=2, length=4)
plt.xlabel('PC1', fontsize=25, weight='bold')
plt.ylabel('PC2', fontsize=25, weight='bold')
#plt.close()
plt.scatter(X_atoms_test[:,0], X_atoms_test[:,1], alpha = 0.2, c = 'red')
plt.savefig("Mix_train_gen_atom_dist_{}Sam.png".format(len(y_train)), bbox_inches='tight', dpi=300)
####

# PCA1 vs. PCA2 Bonds gen and train
plt.close()
fig, ax = plt.subplots(figsize =(8, 6))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
plt.scatter(X_bonds_train[:,0], X_bonds_train[:,1], alpha = 0.2, c = 'blue')
ax.tick_params(width=2, length=4)
plt.xlabel('PC1', fontsize=25, weight='bold')
plt.ylabel('PC2', fontsize=25, weight='bold')
#plt.close()
plt.scatter(X_bonds_test[:,0], X_bonds_test[:,1], alpha = 0.2, c = 'red')
plt.savefig("Mix_train_gen_bonds_dist_{}Sam.png".format(len(preds)), bbox_inches='tight', dpi=300)

### concat. latent vectors ###
X_Concat_train =  np.concatenate ([X_bonds_train_, X_atoms_train_], axis=1)
X_Concat_test  =  np.concatenate ([X_bonds_test_, X_atoms_test_], axis=1)
pca_2 = PCA(n_components = 2)
X_concat_train_pca = pca_2.fit_transform(X_Concat_train)
X_concat_test_pca = pca_2.transform(X_Concat_test)

# PCA1 vs. cv gen. and train
plt.close()
fig, ax = plt.subplots(figsize =(10, 6))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
preds_rev = [max(preds)-i for i in preds]
plt.scatter(X_concat_test_pca[:, 0], preds)
#plt.legend(fontsize=12)
ax.tick_params(width=2, length=4)
plt.xlabel('PC1', fontsize=25, fontweight='bold')
plt.ylabel('Cv', fontsize=25, fontweight='bold')
plt.scatter(X_concat_train_pca[:, 0], y_train)
plt.savefig("genvstrain_Concat_pc1vscv.png", bbox_inches='tight', dpi=300)

# PCA2 vs. cv gen and train
plt.close()
fig, ax = plt.subplots(figsize =(10, 6))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
preds_rev = [max(preds)-i for i in preds]
plt.scatter(X_concat_test_pca[:, 1], preds, c='red')
#plt.legend(fontsize=12)
ax.tick_params(width=2, length=4)
plt.xlabel('PC2', fontsize=25, fontweight='bold')
plt.ylabel('Cv', fontsize=25, fontweight='bold')
plt.scatter(X_concat_train_pca[:, 1], y_train, c='blue')
plt.savefig("genvstrain_Concat_pc2vscv.png", bbox_inches='tight', dpi=300)

# PCA1 vs. PCA2 gen and train
plt.close()
fig, ax = plt.subplots(figsize =(8, 6))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
plt.scatter(X_concat_train_pca[:, 0], X_concat_train_pca[:, 1], alpha=0.2, c='blue')
ax.tick_params(width=2, length=4)
plt.xlabel('PC1', fontsize=25, weight='bold')
plt.ylabel('PC2', fontsize=25, weight='bold')
#plt.close()
plt.scatter(X_concat_test_pca[:, 0], X_concat_test_pca[:, 1], alpha=0.2, c='red')
plt.savefig("genvstrain_concat_pc1vspc2.png".format(len(preds)), bbox_inches='tight', dpi=300)
"""

colors = ['navy', 'mediumblue','blue', 'cornflowerblue', 'lightsteelblue', 'lavender',
          'salmon', 'lightcoral', 'orangered', 'darkred']
colors = ['darkblue', 'lightsteelblue', 'orangered', 'darkred']
"""
group_names = np.array(["Cv<{}".format(np.round(Qs[0])), 
                            "{}<Cv<{}".format(np.round(Qs[0]), np.round(Qs[1])),
                            "{}<Cv<{}".format(np.round(Qs[1]), np.round(Qs[2])),
                            "{}<Cv<{}".format(np.round(Qs[2]), np.round(Qs[3])),
                            "{}<Cv<{}".format(np.round(Qs[3]), np.round(Qs[4])),
                            "{}<Cv<{}".format(np.round(Qs[4]), np.round(Qs[5])),
                            "{}<Cv<{}".format(np.round(Qs[5]), np.round(Qs[6])),
                            "{}<Cv<{}".format(np.round(Qs[6]), np.round(Qs[7])),
                            "{}<Cv<{}".format(np.round(Qs[7]), np.round(Qs[8])),
                            "{}<Cv".format(np.round(Qs[8]))])

group_names_gen = np.array(["Cv<{}".format(np.round(Qs_gen[0])), 
                            "{}<Cv<{}".format(np.round(Qs_gen[0]), np.round(Qs_gen[1])),
                            "{}<Cv<{}".format(np.round(Qs_gen[1]), np.round(Qs_gen[2])),
                            "{}<Cv<{}".format(np.round(Qs_gen[2]), np.round(Qs_gen[3])),
                            "{}<Cv<{}".format(np.round(Qs_gen[3]), np.round(Qs_gen[4])),
                            "{}<Cv<{}".format(np.round(Qs_gen[4]), np.round(Qs_gen[5])),
                            "{}<Cv<{}".format(np.round(Qs_gen[5]), np.round(Qs_gen[6])),
                            "{}<Cv<{}".format(np.round(Qs_gen[6]), np.round(Qs_gen[7])),
                            "{}<Cv<{}".format(np.round(Qs_gen[7]), np.round(Qs_gen[8])),
                            "{}<Cv".format(np.round(Qs_gen[8]))])
"""
group_names = np.array(["Cv<{}".format(np.round((Qs[1]+Qs[2])/2)), 
                            "{}<Cv<{}".format(np.round((Qs[1]+Qs[2])/2), np.round(Qs[4])),
                            "{}<Cv<{}".format(np.round(Qs[4]), np.round((Qs[6]+Qs[7])/2)),
                            "{}<Cv".format(np.round((Qs[6]+Qs[7])/2))])
#group_names_gen = group_names
group_names_gen = np.array(["Cv<{}".format(np.round((Qs_gen[1]+Qs_gen[2])/2)),
                            "{}<Cv<{}".format(np.round((Qs_gen[1]+Qs_gen[2])/2), np.round(Qs_gen[4])),
                            "{}<Cv<{}".format(np.round(Qs_gen[4]), np.round((Qs_gen[6]+Qs_gen[7])/2)),
                            "{}<Cv".format(np.round((Qs_gen[6]+Qs_gen[7])/2))])

target_ids = range(0, 4)
# pc1 vs. pc2 concat 
X_Concat_train =  np.concatenate ([X_bonds_train_, X_atoms_train_], axis=1)
X_Concat_test  =  np.concatenate ([X_bonds_test_, X_atoms_test_], axis=1)
pca_2 = PCA(n_components = 3)
X_concat_train_pca = pca_2.fit_transform(X_Concat_train)
X_concat_test_pca = pca_2.transform(X_Concat_test)


plt.close()
fig, ax = plt.subplots(figsize =(10, 5))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
for i, c, label in zip(target_ids, colors, group_names):
            plt.scatter(X_concat_train_pca[y_class == i, 0],
                        X_concat_train_pca[y_class == i, 1],
                        alpha=0.5, c=c, label=label)
plt.legend(fontsize=12)
ax.tick_params(width=2, length=4)
plt.xlabel('PC1', fontsize=25, fontweight='bold')
plt.ylabel('PC2', fontsize=25, fontweight='bold')
plt.savefig("train_conc_dist_pca.png", bbox_inches='tight', dpi=300)


plt.close()
fig, ax = plt.subplots(figsize =(10, 5))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
for i, c, label in zip(target_ids, colors, group_names_gen):
            plt.scatter(X_concat_test_pca[y_class_val == i, 0],
                        X_concat_test_pca[y_class_val == i, 1],
                        alpha=0.5, c=c, label=label)
plt.legend(fontsize=12)
ax.tick_params(width=2, length=4)
plt.xlabel('PC1', fontsize=25, fontweight='bold')
plt.ylabel('PC2', fontsize=25, fontweight='bold')
plt.savefig("test_conc_dist_pca.png", bbox_inches='tight', dpi=300)

plt.close()
fig, ax = plt.subplots(figsize =(10, 5))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
for i, c, label in zip(target_ids, colors, group_names):
            plt.scatter(X_concat_train_pca[y_class == i, 0],
                        X_concat_train_pca[y_class == i, 1],
                        alpha=0.5, c=c, label=label)
plt.legend(fontsize=12)
ax.tick_params(width=2, length=4)
plt.xlabel('PC1', fontsize=25, fontweight='bold')
plt.ylabel('PC2', fontsize=25, fontweight='bold')
plt.savefig("train_conc_dist_pca.png", bbox_inches='tight', dpi=300)


perplexities = [20, 30, 40, 45, 50, 60, 100, 120]
perplexities = [40, 45, 50]
seed = 100
pbar = ProgressBar()

n_iter = 10000

### concat. latent vectors ###
X_Concat_train =  np.concatenate ([X_bonds_train_, X_atoms_train_], axis=1)
X_Concat_test  =  np.concatenate ([X_bonds_test_, X_atoms_test_], axis=1)
pca_2 = PCA(n_components=3)
X_concat_train_pca = pca_2.fit_transform(X_Concat_train)
X_concat_test_pca = pca_2.transform(X_Concat_test)

### tsne of combined atom and bond matrices ###
X_Concat_train_tsne = TSNE(verbose=1, learning_rate=100, min_grad_norm=1e-50,
				   n_iter=n_iter, n_components=2, perplexity=48, 
                                   random_state=150, angle=0.5, square_distances=True,
				  init='pca').fit_transform(X_concat_train_pca)
X_Concat_test_tsne = TSNE(verbose=1, learning_rate=100, n_iter=n_iter, min_grad_norm=1e-50,
		          n_components=2,   perplexity=48, random_state=150, angle=0.5, n_jobs=-1, square_distances=True,  
		          init='pca').fit_transform(X_concat_test_pca)
	
###
"""
# using .pickle to save mapped vectors	
with open('./TSNE_variables.pickle', 'wb') as f:
		pickle.dump((X_Concat_train_tsne, X_Concat_test_tsne), f)
	
with open('./TSNE_variables.pickle', 'rb') as f:
		X_Concat_train_tsne, X_Concat_test_tsne = pickle.load(f)
"""

# using .csv file to save mapped vectors
tsne_components = pd.DataFrame()
tsne_components['SMILES'] = SMILES_gantrain
tsne_components['cv'] = y_train
tsne_components['class'] = y_class
tsne_components['train_tsne1'] = np.array (X_Concat_train_tsne)[:, 0]
tsne_components['train_tsne2'] = np.array (X_Concat_train_tsne)[:, 1]
tsne_components.to_csv('./TSNE_variables_train1.csv')

tsne_components = pd.DataFrame()
tsne_components['SMILES'] = gen_smiles
tsne_components['pred_cv'] = preds
tsne_components['class'] = y_class_val
tsne_components['test_tsne1'] = np.array (X_Concat_test_tsne)[:, 0]
tsne_components['test_tsne2'] = np.array (X_Concat_test_tsne)[:, 1]
tsne_components.to_csv('./TSNE_variables_test1.csv')


tsne_components = pd.read_csv('./TSNE_variables_train1.csv')
X_Concat_train_tsne1 = np.array (tsne_components['train_tsne1'])
X_Concat_train_tsne2 = np.array (tsne_components['train_tsne2'])

tsne_components = pd.read_csv('./TSNE_variables_test1.csv')
X_Concat_test_tsne1 = np.array (tsne_components['test_tsne1'])
X_Concat_test_tsne2 = np.array (tsne_components['test_tsne2'])

X_Concat_train_tsne = np.vstack ((X_Concat_train_tsne1, X_Concat_train_tsne2))
print ('train tsne', X_Concat_train_tsne.shape)
X_Concat_test_tsne = np.vstack ((X_Concat_test_tsne1, X_Concat_test_tsne2))
print ('test tsne', X_Concat_test_tsne.shape)
# tsne1 vs. tsne 2 train
X_Concat_train_tsne = X_Concat_train_tsne.T
X_Concat_test_tsne = X_Concat_test_tsne.T
plt.close()
fig, ax = plt.subplots(figsize=(10, 10))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
for i, c, label in zip(target_ids, colors, group_names):
	    plt.scatter(X_Concat_train_tsne[y_class == i, 0], 
			X_Concat_train_tsne[y_class == i, 1], 
			alpha=1, c=c, label=label)	
plt.xlim(-50, 50)
plt.ylim(-50, 50)
id_sample = 13
class_sample = 0
plt.scatter(X_Concat_train_tsne[y_class == class_sample, 0][id_sample],
                        X_Concat_train_tsne[y_class == class_sample, 1][id_sample],
                        alpha=1, c='black', marker='X', s=30)
print ('train SMILES from class {} and id {}'.format(class_sample, id_sample), 
		SMILES_gantrain[y_class == class_sample][id_sample])
print ('train Cv from class {} and id {}'.format(class_sample, id_sample),
		y_train [y_class == class_sample][id_sample])

#plt.legend(fontsize=12, bbox_to_anchor=(0, 2))
ax.tick_params(width=2, length=4)
plt.xlabel('t-SNE1', fontsize=25, fontweight='bold')
plt.ylabel('t_SNE2', fontsize=25, fontweight='bold')
plt.savefig("train_concat_dist_tsne_p50s100.png", bbox_inches='tight', dpi=300)


# tsne1 vs. tsne2 gen.
plt.close()
fig, ax = plt.subplots(figsize =(10, 10))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
for i, c, label in zip(target_ids, colors, group_names_gen):
	    plt.scatter(X_Concat_test_tsne[y_class_val == i, 0], 
			X_Concat_test_tsne[y_class_val == i, 1], 
			alpha=1, c=c, label=label)
"""
# left cluster
plt.scatter(X_Concat_test_tsne[y_class_val == 3, 0][20],
                        X_Concat_test_tsne[y_class_val == 3, 1][20],
                        alpha=1, c='yellow', marker='*', s=30)
# left cluster
plt.scatter(X_Concat_test_tsne[y_class_val == 3, 0][0],
                        X_Concat_test_tsne[y_class_val == 3, 1][0],
                        alpha=1, c='green', marker='*', s=30)
# right, blue boundary
plt.scatter(X_Concat_test_tsne[y_class_val == 3, 0][10],
                        X_Concat_test_tsne[y_class_val == 3, 1][10],
                        alpha=1, c='black', marker='*', s=30)
# left cluster
plt.scatter(X_Concat_test_tsne[y_class_val == 3, 0][150],
                        X_Concat_test_tsne[y_class_val == 3, 1][150],
                        alpha=1, c='green', marker='x', s=30)
# left cluster
plt.scatter(X_Concat_test_tsne[y_class_val == 3, 0][160],
                        X_Concat_test_tsne[y_class_val == 3, 1][160],
                        alpha=1, c='black', marker='X', s=30)
"""
"""
id_sample = 28
class_sample = 0
plt.scatter(X_Concat_test_tsne[y_class_val == class_sample, 0][id_sample],
		X_Concat_test_tsne[y_class_val == class_sample, 1][id_sample],	
		alpha=1, c='black', marker='X', s=30)
gen_smiles = np.array(gen_smiles)
preds = np.array(preds)
print ('test SMILES from class {} and id {}'.format(class_sample, id_sample),
		gen_smiles[y_class_val == class_sample][id_sample])
print ('test Cv from class {} and id {}'.format(class_sample, id_sample),
                preds [y_class_val == class_sample][id_sample])
"""
#plt.legend(fontsize=12, bbox_to_anchor=(0, 2))
#plt.legend(fontsize=12)
plt.xlim(-55, 55)
plt.ylim(-40, 40)
ax.tick_params(width=2, length=4)
plt.xlabel('t-SNE1', fontsize=25, fontweight='bold')
plt.ylabel('t_SNE2', fontsize=25, fontweight='bold')
plt.savefig("gen_Concat_dist_tsne_p50s100.png", bbox_inches='tight', dpi=300)


logp_train = []
for s in SMILES_gantrain:
    m = AllChem.MolFromSmiles(s)
    logp_train.append(logp.MolLogP(m))

logp_test = []
for s in gen_smiles:
    m = AllChem.MolFromSmiles(s)
    logp_test.append(logp.MolLogP(m))

logp_train = np.array(logp_train)
logpclass_train = logp_train
logpclass_train = np.where(logp_train <= 2, 0, logpclass_train)
logpclass_train = np.where(logp_train > 2, 1, logpclass_train)
print ('logptrain', logp_train)
print ('logpclass_train', logpclass_train)
group_names = np.array (["logp < 2", "logp > 2"])
target_ids = range(0,2)
colors = ['red', 'blue']
# tsne1 vs. tsne2 atoms train
plt.close()
rc('font', weight='bold')
fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
for i, c, label in zip(target_ids, colors, group_names):
    plt.scatter(X_Concat_train_tsne[logpclass_train == i, 0], 
                X_Concat_train_tsne[logpclass_train == i, 1], 
                c=c, label=label)
plt.legend(fontsize=12)
ax.tick_params(width=2, length=4)
plt.xlabel('t-SNE1', fontsize=25, fontweight='bold')
plt.ylabel('t-SNE2', fontsize=25, fontweight='bold')
plt.savefig("train_atom_dist_tsne_logp.png", bbox_inches='tight', dpi=300)

logp_test = np.array(logp_test)
logpclass_test = logp_test
logpclass_test = np.where(logp_test <= 2, 0, logpclass_test)
logpclass_test = np.where(logp_test > 2, 1, logpclass_test)
print ('logp_test', logp_test)
print ('logpclass_test', logpclass_test)
group_names = np.array (["logp < 2", "logp > 2"])
target_ids = range(0,2)
colors = ['red', 'blue']
# tsne1 vs. tsne2 atoms test
plt.close()
rc('font', weight='bold')
fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(axis='both', which='major', labelsize=20)
mpl.rcParams['axes.linewidth'] = 3.5
for i, c, label in zip(target_ids, colors, group_names):
    plt.scatter(X_Concat_test_tsne[logpclass_test == i, 0],
                X_Concat_test_tsne[logpclass_test == i, 1],
                c=c, label=label)
plt.legend(fontsize=12)
ax.tick_params(width=2, length=4)
plt.xlabel('t-SNE1', fontsize=25, fontweight='bold')
plt.ylabel('t-SNE2', fontsize=25, fontweight='bold')
plt.savefig("test_atom_dist_tsne_logp.png", bbox_inches='tight', dpi=300)


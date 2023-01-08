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

import numpy as np
from numpy import ndarray

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from scipy.stats import truncnorm

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
randomtf.set_seed(10)
os.environ['PYTHONHASHSEED'] = '10'
np.random.seed(420)
random.seed(123450)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3667)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, gpu_options=gpu_options)
#tf.set_random_seed(1234)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

tf.config.list_physical_devices('GPU')

""" reading and preprocessing data"""
with open('./../data/trainingsets/highcv_qm9/image_train.pickle', 'rb') as f:
    X_smiles_train0, SMILES_train0, X_atoms_train, X_bonds_train, y_train0 = pickle.load(f)
    
with open('./../data/trainingsets/highcv_qm9/image_test.pickle', 'rb') as f:
    X_smiles_test0, SMILES_test0, X_atoms_val, X_bonds_val, y_test0 = pickle.load(f)

with open('./../data/trainingsets/60000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
tokenizer[0] = ' '

with open('./../data/trainingsets/highcv_qm9/train_GAN.pickle', 'rb') as f:
    X_smiles_gantrain0, SMILES_gantrain0, _, __, cv_gantrain0 = pickle.load(f)

with open('./../data/trainingsets/highcv_DFT_trans1/highcv_DFT_test.pickle', 'rb') as f:
    X_smiles_test_dft, SMILES_test_dft, y_test_dft = pickle.load(f)
with open('./../data/trainingsets/highcv_DFT_trans1/highcv_DFT_train.pickle', 'rb') as f:
    X_smiles_train_dft, SMILES_train_dft, y_train_dft = pickle.load(f)
   
with open('./../data/trainingsets/highcv_DFT_trans1/highcv_DFT.pickle', 'rb') as f:
    X_smiles_gantrain_dft, SMILES_gantrain_dft, cv_gantrain_dft = pickle.load(f)

X_smiles_gantrain_ = []
for i in X_smiles_gantrain0:
  X_smiles_gantrain_.append(i)
X_smiles_gantrain0 = np.array (X_smiles_gantrain_)
print (X_smiles_gantrain0.shape)

cv_gantrain_ = []
for i in cv_gantrain0:
  cv_gantrain_.append(i)
cv_gantrain0 = np.array (cv_gantrain_)
print (cv_gantrain0.shape)

X_smiles_train_dft = X_smiles_train_dft.reshape((X_smiles_train_dft.shape[0], 35, 23, 1))
X_smiles_test_dft = X_smiles_test_dft.reshape((X_smiles_test_dft.shape[0], 35, 23, 1))
X_smiles_gantrain_dft = X_smiles_gantrain_dft.reshape((X_smiles_gantrain_dft.shape[0], 35, 23, 1))

y_train = np.concatenate([y_train0, y_train_dft], axis=0)
X_smiles_train = np.concatenate([X_smiles_train0, X_smiles_train_dft], axis=0)
SMILES_train = np.concatenate([SMILES_train0, SMILES_train_dft], axis=0)

y_val = np.concatenate([y_test0, y_test_dft], axis=0)
X_smiles_val = np.concatenate([X_smiles_test0, X_smiles_test_dft], axis=0)
SMILES_val = np.concatenate([SMILES_test0, SMILES_test_dft], axis=0)

cv_gantrain = np.concatenate([cv_gantrain0, cv_gantrain_dft], axis=0)
X_smiles_gantrain = np.concatenate([X_smiles_gantrain0, X_smiles_gantrain_dft], axis=0)
SMILES_gantrain = np.concatenate([SMILES_gantrain0, SMILES_gantrain_dft], axis=0)

# Subsampling has been done in the data preprocesses
print ('X_smiles_train shape: ', X_smiles_train.shape)
print ('X_smiles_test shape: ', X_smiles_val.shape)
#print ('last SMILES train: ', SMILES_train[-1])

"""
## Outlier removal 1.5*IQR rule
# Train samples
IQR = - np.quantile(y_train0, 0.25) + np.quantile(y_train0, 0.75)
lower_bound, upper_bound = np.quantile(y_train0, 0.25) - 1.5 * IQR, np.quantile(y_train0, 0.75) + 1.5 * IQR
idx = np.where((y_train0 >= lower_bound) & (y_train0 <= upper_bound))

y_train = y_train0[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]

# Test samples
IQR = - np.quantile(y_val0, 0.25) + np.quantile(y_val0, 0.75)
lower_bound, upper_bound = np.quantile(y_val0, 0.25) - 1.5 * IQR, np.quantile(y_val0, 0.75) + 1.5 * IQR
idx = np.where((y_val0 >= lower_bound) & (y_val0 <= upper_bound))

y_val = y_val0[idx]
X_smiles_val = X_smiles_val[idx]
X_atoms_val = X_atoms_val[idx]
X_bonds_val = X_bonds_val[idx]
"""

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
s_min_norm, s_max_norm = 20.939, 55

y_val   = (y_val -   s_min_norm) / (s_max_norm - s_min_norm)
y_train = (y_train - s_min_norm) / (s_max_norm - s_min_norm)
cv_gantrain = (cv_gantrain - s_min_norm) / (s_max_norm - s_min_norm)
print ("min and max dataset and val normalized", s_min, s_max, np.min(y_val), np.max(y_val))
print ("min and max dataset and train normalized", s_min, s_max, np.min(y_train), np.max(y_train))
print ("min and max used for normalization: ", s_min_norm, s_max_norm)

""" models definition and extracting pretrained encoder and decoder """
encoder = load_model('./../data/nns/encoder_newencinp_trans.h5')
decoder = load_model('./../data/nns/decoder_newencinp_trans.h5')

class Config:
    
    def __init__(self):
        self.Filters = [256, 128, 64]
        self.genFilters = [128, 128, 128]
        self.upFilters = [(2, 2), (2, 2), (2, 2)]
        
config = Config()

## Generator 
z = Input(shape = (128, ))
y = Input(shape = (1, ))

h = Concatenate(axis = 1)([z, y])
h = Dense(1 * 1 * 128)(h)
R1 = Reshape([1, 1, 128])(h)
R2 = Reshape([1, 1, 128])(h)

for i in range(3):
    R1 = UpSampling2D(size = config.upFilters[i])(R1)
    C1 = Conv2D(filters = config.genFilters[i], 
               kernel_size = 2, 
               strides = 1, 
               padding = 'same')(R1)
    B1 = BatchNormalization()(C1)
    R1 = LeakyReLU(alpha=0.2)(B1)

for i in range(3):
    R2 = UpSampling2D(size = config.upFilters[i])(R2)
    C2 = Conv2D(filters = config.genFilters[i], 
               kernel_size = 2, 
               strides = 1, 
               padding = 'same')(R2)
    B2 = BatchNormalization()(C2)
    R2 = LeakyReLU(alpha=0.2)(B2)
    
R1 = Conv2D(1,
            kernel_size = 3,
            strides = 1,
            padding = 'valid',
            activation = 'tanh')(R1)
R2 = Conv2D(1,
            kernel_size = 3,
            strides = 1,
            padding = 'valid',
            activation = 'tanh')(R2)

generator = Model([z, y], [R1, R2])
print (generator.summary())

## Discriminator 
inp1 = Input(shape = [6, 6, 1])
inp2 = Input(shape = [6, 6, 1])

X1 = Concatenate()([inp1, inp2])
X = Flatten()(X1)
y2 = Concatenate(axis = 1)([X, y])
for i in range(3):
		y2 = Dense(64, activation = 'relu')(y2)
		y2 = LeakyReLU(alpha = 0.2)(y2)
		y2 = Dropout(0.2)(y2)

O_dis = Dense(1, activation = 'sigmoid')(y2)


discriminator = Model([inp1, inp2, y], O_dis)
discriminator.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 5e-7, beta_1 = 0.5))
print (discriminator.summary()) 

## Regressor
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

# Training the Regressor 
# latent vectors from trained Encoder, 
# last output of Encoder is concat. (O1, O2)
train_atoms_embedding, train_bonds_embedding, _ = encoder.predict([X_smiles_gantrain])
atoms_embedding, bonds_embedding, _ = encoder.predict([X_smiles_gantrain])
atoms_val, bonds_val, _ = encoder.predict([X_smiles_val])

#try:
#regressor =     load_model('./../data/nns/regressor_trans.h5')
#regressor_top = load_model('./../data/nns/regressor_top_trans.h5')
regressor =     load_model('./../data/nns/regressor_55max.h5')
regressor_top = load_model('./../data/nns/regressor_top_55max.h5')
print (".h5 was read")
"""
except:
    print ("no .h5 available")
    regressor.compile(loss = 'mse', optimizer = Adam(1e-5))
    pass
"""
#regressor.compile(loss = 'mse', optimizer = Adam(1e-6))
regressor.trainable = True
regressor_top.trainable = True
print (regressor.summary())

history = regressor.fit([atoms_embedding, bonds_embedding], 
              cv_gantrain,
              validation_data = ([atoms_val,
                                  bonds_val],
                                 y_val),
              batch_size = 8,
              epochs = 20,
              verbose = 1)

# keep test data unseen
"""
history = regressor.fit([atoms_embedding, bonds_embedding],
              cv_gantrain,
              batch_size = 128,
              epochs = 1,
              verbose = 1)
"""
# Validating the regressor
# Train

pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for train data: {}'.format(r2_score(cv_gantrain, pred.reshape([-1]))))
mse_train_normalized = mean_squared_error(cv_gantrain, pred.reshape([-1]))
mse_train = mse_train_normalized * ((s_max_norm - s_min_norm)**2)
print ('norm. train MSE: ', mse_train_normalized, ', train MSE: ', mse_train)
print ('norm. train RMSE: ', mse_train_normalized**0.5, ', train RMSE: ', mse_train**0.5)
print ('prediction on train: ', pred)
print ('True train: ', cv_gantrain)
# Test
pred = regressor.predict([atoms_val, bonds_val])
print('Current R2 on Regressor for test data: {}'.format(r2_score(y_val, pred.reshape([-1]))))
mse_test_normalized = mean_squared_error(y_val, pred.reshape([-1]))
mse_test = mse_test_normalized * ((s_max_norm - s_min_norm)**2)
print ('norm. test MSE: ', mse_test_normalized, 'test MSE: ', mse_test)
print ('norm. test RMSE: ', mse_test_normalized**0.5, 'test RMSE: ', mse_test**0.5)
print ("prediction on test: ", pred )
print ("True test values: ", y_val)

# Saving the currently trained models
#regressor.save('./../data/nns/regressor_trans.h5')
#regressor_top.save('./../data/nns/regressor_top_trans.h5')

"""
# save the losses 
with open ('regressor_loss_100_150.csv', 'w') as f:
    for key in history.history.keys():
        f.write("%s,%s\n"%(key,history.history[key]))

# plot history for loss
plt.close()
plt.plot(history.history['loss'])
plt.title('Regressor loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("R_loss_0_100.png", dpi=300)
"""

## Combined model 
def build_combined(z, y,
                   regressor,
                   regressor_top,
                   discriminator,
                   encoder,
                   decoder):
    discriminator.trainable = False
    regressor_top.trainable = False
    regressor.trainable = False
    encoder.trainable = False
    decoder.trainable = False
    
    atoms_emb, bonds_emb = generator([z, y])
    dec_embedding = Concatenate()([atoms_emb, bonds_emb])
    
    softmax_smiles, _ = decoder([dec_embedding])
    argmax_smiles = argmax (softmax_smiles, axis=2)
    argmax_smiles = Reshape([35])(argmax_smiles)
    smiles = one_hot(argmax_smiles, depth=23)
    smiles = Reshape([35, 23, 1])(smiles)
    latent_encoder_atom, latent_encoder_bond, _ = encoder ([smiles])
    
    y_pred = regressor([latent_encoder_atom, latent_encoder_bond])
    valid = discriminator([atoms_emb, bonds_emb, y])

    combined = Model([z, y], [valid, y_pred])

    combined.compile(loss = ['binary_crossentropy',
                             'mse'], 
                     loss_weights = [1.0, 25.0], 
                     optimizer = Adam(5e-7, beta_1 = 0.5))
    return combined

combined = build_combined(z, y,
                          regressor,
                          regressor_top,
                          discriminator,
                          encoder,
                          decoder)

""" Training RCGAN """
# loading pretrained models
#regressor = load_model    ('./../data/nns/regressor.h5')
#regressor_top = load_model('./../data/nns/regressor_top.h5')
#generator = load_model    ('./../data/nns/generator_55max.h5')
#discriminator= load_model ('./../data/nns/discriminator_55max.h5')
#generator.load_weights('./../data/nns/keep/generator.h5')
#discriminator.load_weights('./../data/nns/keep/discriminator.h5')
generator = load_model    ('./../data/nns/generator_trans.h5')
discriminator= load_model ('./../data/nns/discriminator_trans.h5')


regressor_top.trainable = False
regressor.trainable = False

# SMILES related information
max_gen_atoms = 9
bond_max = 9
MAX_NB_WORDS = 23
MAX_SEQUENCE_LENGTH = 35

"""
epochs = 5
batch_size = 16
batches = cv_gantrain.shape[0] // batch_size
threshold = 0.3 # defining accurate samples
reinforce_n = 50 # 5*reinforce_n = fake sampling
reinforce_sample = 1000 # how many samples generated for Reinforcement

# variable for storing generated data
G_Losses = []
D_Losses = []
R_Losses = []
D_Losses_real = []
D_Losses_fake = []

for e in range(epochs):
    start = time.time()
    D_loss = []
    G_loss = []
    R_loss = []
    D_loss_real = []
    D_loss_fake = []
    
    for b in range(batches):
        
        regressor_top.trainable = False
        regressor.trainable = False

        idx = np.arange(b * batch_size, (b + 1) * batch_size)
        # rearrange the samples 
        idx = np.random.choice(idx, batch_size, replace = False)
        
        x_smiles_gantrain = X_smiles_gantrain[idx] 
        batch_y = cv_gantrain[idx]
        
        batch_z = np.random.normal(0, 1, size = (batch_size, 128))
        
        atoms_embedding, bonds_embedding, _ = encoder.predict([x_smiles_gantrain])
        dec_embedding = np.concatenate([atoms_embedding, bonds_embedding], axis = -1)
        
        gen_atoms_embedding, gen_bonds_embedding = generator.predict([batch_z, batch_y])
        
        gen_dec_embedding = np.concatenate([gen_atoms_embedding, gen_bonds_embedding], axis = -1)
        softmax_smiles = decoder.predict(gen_dec_embedding)[0]
        argmax_smiles = np.argmax(softmax_smiles, axis = 2)
        smiles = to_categorical(argmax_smiles, num_classes=23)
        SHAPE = list(smiles.shape) + [1]
        smiles = smiles.reshape(SHAPE)
        latent_encoder_atom, latent_encoder_bond, _ = encoder.predict([smiles])
        gen_pred = regressor.predict([latent_encoder_atom, latent_encoder_bond]).reshape([-1])
        
        regressor.trainable = True
        r_loss = regressor.train_on_batch([atoms_embedding, bonds_embedding], batch_y)
        R_loss.append(r_loss)
        regressor.trainable = False

        discriminator.trainable = True
        # original was 3!
        d = 3
        #if b<100:
        #    d=1
        for _ in range(d):
            d_loss_real = discriminator.train_on_batch([atoms_embedding, bonds_embedding, batch_y],
                                                       [1 * np.ones((batch_size, 1))])
            d_loss_fake = discriminator.train_on_batch([gen_atoms_embedding, gen_bonds_embedding, batch_y],
                                                       [np.zeros((batch_size, 1))])

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        D_loss.append(d_loss)
        D_loss_real.append (d_loss_real)
        D_loss_fake.append (d_loss_fake)
        discriminator.trainable = False
        
        regressor_top.trainable = False
        regressor.trainable = False

        #for _ in range(d):
        g_loss = combined.train_on_batch([batch_z, batch_y], [1 * np.ones((batch_size, 1)), batch_y])
        G_loss.append(g_loss)
    
    D_Losses.append(np.mean(D_loss))
    D_Losses_real.append(np.mean(D_loss_real))
    D_Losses_fake.append(np.mean(D_loss_fake))
    G_Losses.append(np.mean(G_loss))
    R_Losses.append(np.mean(R_loss))
    
    print('====')
    print('Current epoch: {}/{}'.format((e + 1), epochs))
    print ('D Loss Real: {}'.format(np.mean(D_loss_real)))
    print ('D Loss Fake: {}'.format(np.mean(D_loss_fake)))
    print('D Loss: {}'.format(np.mean(D_loss)))
    print('G Loss: {}'.format(np.mean(G_loss)))
    print('R Loss: {}'.format(np.mean(R_loss)))
    print('====')
    print()

    
    # Reinforcement
    gen_error = []
    gen_smiles = []
    gen_valid_smiles = []
    gen_X_atoms = []
    gen_X_bonds = []
    predcv_AE_latent = []
    embeddings = []
    sample_ys = []
    valid_smiles_index = []
    for _ in range(reinforce_sample):
        sample_y = np.random.uniform(s_min_dataset, s_max_dataset, size = [1,])
        sample_y = np.round(sample_y, 4)
        sample_y = (sample_y - s_min_norm) / (s_max_norm - s_min_norm)
        sample_ys.append(sample_y)

        sample_z = np.random.normal(0, 1, size = (1, 128))

        sample_atoms_embedding, sample_bonds_embedding = generator.predict([sample_z, sample_y])
        embeddings.append((sample_atoms_embedding, sample_bonds_embedding))
        
        dec_embedding = np.concatenate([sample_atoms_embedding, sample_bonds_embedding], axis = -1)
        softmax_smiles = decoder.predict(dec_embedding)[0]
        argmax_smiles = np.argmax(softmax_smiles, axis = 2).reshape([-1])
        smiles = to_categorical(argmax_smiles, num_classes=23)
        SHAPE = [1] + list(smiles.shape) + [1]
        smiles = smiles.reshape(SHAPE)
        c_smiles = ''
        for s in argmax_smiles:
            c_smiles += tokenizer[s]
        c_smiles = c_smiles.rstrip()
        
        if _==0:
            #print ("Gen. sample Reinforce center from Decoder", smiles)
            print ('"   "   converted to SMILES"',c_smiles)
        gen_smiles.append(c_smiles)
        latent_encoder_atom, latent_encoder_bond, _ = encoder.predict([smiles])
        reg_pred = regressor.predict([latent_encoder_atom, latent_encoder_bond])
        
        pred, desire = reg_pred[0][0], sample_y[0]
        gen_error.append(np.abs((pred - desire) / desire))

        
    gen_error = np.asarray(gen_error)
    # two validity defined: 
    # without sanitizing: valid 0    
    valid = 0
    valid0 = 0
    idx_ = []
    idx0_ = []
    for iter_, smiles in enumerate(gen_smiles):
        if ' ' in smiles[:-1]:
            continue
        m  = Chem.MolFromSmiles(smiles[:-1], sanitize=True)
        m0 = Chem.MolFromSmiles(smiles[:-1], sanitize=False)
        if m0 is not None:
            valid0 += 1
            idx0_.append(iter_)
        if m is not None:
            if len(construct_atomic_number_array(m)) <= 11:
                valid += 1
                idx_.append(iter_)
                try:
                    gen_smiles [iter_] = Chem.MolToSmiles(m)
                    print (Chem.MolToSmiles(m))
                    print ("Hc_des", sample_ys[iter_])
                    print ("error", gen_error[iter_])
                except:
                    pass
    idx_ = np.asarray(idx_)
    idx0_ = np.asarray(idx0_)

    validity = [gen_smiles[jj] for jj in idx0_ ]
    validity = pd.DataFrame(validity)
    validity = validity.drop_duplicates()

    validity_sanitize = [gen_smiles[jj] for jj in idx_ ]
    validity_sanitize = pd.DataFrame(validity_sanitize)
    validity_sanitize = validity_sanitize.drop_duplicates()

    if (e + 1) % 100 == 0:
        reinforce_n += 10

    # invalid smiles:
    fake_indices1 = np.setdiff1d(np.arange(reinforce_sample), np.asarray(idx_))
    fake_indices2 = np.intersect1d(np.where(gen_error > threshold)[0], idx_)
    fake_indices = np.concatenate ((fake_indices1, fake_indices2))
    fake_indices = np.random.choice(fake_indices, reinforce_n * 5, replace = False)

    real_indices_ = np.intersect1d(np.where(gen_error < threshold)[0], idx_)
    sample_size =  len(real_indices_)
    real_indices = np.random.choice(real_indices_, sample_size, replace = False)
    
    # Activating Reinforcement 
    if e >= 5:
        discriminator.trainable = True
        regressor_top.trainable = False
        regressor.trainable = False
        for real_index in real_indices:
            #real_latent = regressor_top.predict([embeddings[real_index][0], embeddings[real_index][1]])
            _ = discriminator.train_on_batch([embeddings[real_index][0], embeddings[real_index][1], sample_ys[real_index]],
                                             [1 * np.ones((1, 1))])

        for fake_index in fake_indices:
            #fake_latent = regressor_top.predict([embeddings[fake_index][0], embeddings[fake_index][1]])
            _ = discriminator.train_on_batch([embeddings[fake_index][0], embeddings[fake_index][1] , sample_ys[fake_index]],
                                             [np.zeros((1, 1))])
        discriminator.trainable = False

    # ==== #
    try:
        print('Currently valid SMILES (No chemical_beauty and sanitize off): {}'.format(valid0))
        print('Currently valid SMILES Unique (No chemical_beauty and sanitize off): {}'.format(len(validity)))
        print('Currently valid SMILES Sanitized: {}'.format(valid))
        print('Currently valid Unique SMILES Sanitized: {}'.format(len(validity_sanitize)))
        print('Currently satisfying SMILES: {}'.format(len(real_indices_)))
        print('Currently unique satisfying generation: {}'.format(len(np.unique(np.array(gen_smiles)[real_indices_]))))
        print('Gen Sample is: {}, for {}'.format(c_smiles, sample_y))
        print('Predicted val: {}'.format(reg_pred))
        print('====')
        print()
    except:
        pass

    if (e + 1) % 5 == 0:
        plt.close()
        fig, ax = plt.subplots(figsize = (12, 10))
        ax.tick_params(axis='both', which='major', labelsize=30)
        plt.plot(G_Losses, color='blue')
        plt.plot(D_Losses, color='red')
        plt.xlabel('epochs', fontsize=35)
        plt.ylabel('loss', fontsize=35)
        mpl.rcParams['axes.linewidth'] = 2.5
        #plt.plot(R_Losses)
        plt.legend(['G Loss', 'D Loss'], fontsize=30)
        plt.savefig("G_D_losses{}.png".format (e+1))
    n_unique = len(np.unique(np.array(gen_smiles)[real_indices_]))
    n_valid = valid
    if valid > 450 and n_unique > 350:
        print('Criteria has satisified, training has ended')
        break

    end = time.time()
    print ("time for current epoch: ", (end - start))


with open('GAN_loss.pickle', 'wb') as f:
    pickle.dump((G_Losses, D_Losses, R_Losses), f)

# Saving the currently trained models
#regressor.save('regressor.h5')
#regressor_top.save('regressor_top.h5')
generator.save('./../data/nns/generator_trans.h5')
discriminator.save('./../data/nns/discriminator_trans.h5')

##====#

# Generation Study

#regressor = load_model('regressor.h5')
#regressor_top = load_model('regressor_top.h5')
#generator = load_model    ('./../data/nns/generator_new.h5')
#discriminator = load_model('./../data/nns/discriminator_new.h5')

encoder = load_model('./../data/nns/encoder_newencinp_trans.h5')
decoder = load_model('./../data/nns/decoder_newencinp_trans.h5')

# Generation workflow
# 1. Given a desired heat capacity
# 2. Generate 10,000 samples of SMILES embedding
# 3. Select the ones with small relative errors (< 10%)
# 4. Transfer them to SMILES
# 5. Filter out the invalid SMILES
"""
# Generate 500 different values of heat capacities

from progressbar import ProgressBar
N = 50
n_sample = 100

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
# 0
np.random.seed(99)

pbar = ProgressBar()
for hc in pbar(range(n_sample)):
    try:
        # get it back to original of s_min to s_max
        sample_y = np.random.uniform(s_min_dataset, s_max_dataset, size=[1,])
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

        accurate = np.where(gen_errors <= 0.2)[0]
        gen_errors = gen_errors[accurate]
        pred = pred[accurate]

        sample_y = sample_y[accurate]
        sample_atoms_embedding = sample_atoms_embedding[accurate]
        sample_bonds_embedding = sample_bonds_embedding[accurate]

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
        generated_smiles = generated_smiles [accurate]
        all_gen_smiles = []
        idx = []
        for i, smiles in enumerate(generated_smiles):
            all_gen_smiles.append(smiles[:-1])

            if ' ' in smiles[:-1]:
                continue
            #m = Chem.MolFromSmiles(smiles[:-1], sanitize=False)
            m = Chem.MolFromSmiles(smiles[:-1], sanitize=True)
            if m is not None:
                if len(construct_atomic_number_array(m)) <= 11:
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
    ss = Chem.MolToSmiles(Chem.MolFromSmiles(s, sanitize=True))
    gen_smiles[i] = ss

output['SMILES'] = gen_smiles
output['des_cv'] = sample_ys
output['pred_cv'] = preds
output['Err_pred_des'] = gen_error

plt.close()
plt.hist(gen_error)
plt.savefig("gen_error_hist.png")

output = pd.DataFrame(output)
output.reset_index(drop = True, inplace = True)
output.to_csv ('./../experiments/regular/trans1_20screen.csv', index=False)
## Statistics  (# pred=True value, Des=prediction)

# total # of samples
N = len(gen_error)
# Explained Variance R2 from sklearn.metrics.explained_variance_score
explained_variance_R2_pred_des = explained_variance_score(output['des_cv'], output['pred_cv'])
print ("explained_varice_R2_pred_des", explained_variance_R2_pred_des)
rsquared = r2_score (output['des_cv'], output['pred_cv'])
print ("r squared r**2", rsquared)
# mean absolute error 
MAE_pred_des = mean_absolute_error(output['pred_cv'], output['des_cv'])
print ("MAE_pred_des", MAE_pred_des)
# Fractioned MAE, more normalized
Fractioned_MAE_pred_des = 0
for pred, des in zip(output['pred_cv'], output['des_cv']):
    Fractioned_MAE_pred_des = Fractioned_MAE_pred_des +  abs(des-pred)/des
Fractioned_MAE_pred_des = Fractioned_MAE_pred_des/N
print ("Fractioned MAE_pred_des", Fractioned_MAE_pred_des)

# root mean squared error (RMSE), sqrt(sklearn ouputs MSE)
RMSE_pred_des = mean_squared_error(output['pred_cv'], output['des_cv'])**0.5
print ("RMSE_pred_des", RMSE_pred_des)

Fractioned_RMSE_pred_des = 0
for pred, des in zip(output['pred_cv'], output['des_cv']):
    Fractioned_RMSE_pred_des = Fractioned_RMSE_pred_des + ((des-pred)/des)**2
Fractioned_RMSE_pred_des = (Fractioned_RMSE_pred_des/N)**0.5
print ("Fractioned_RMSE_pred_des", Fractioned_RMSE_pred_des)

#output = pd.DataFrame(output)
# do not drop duplicate
output2 = output.drop_duplicates(['SMILES'])
#gen_atoms_embedding = np.array(gen_atoms_embedding)
#gen_bonds_embedding = np.array(train_atoms_embedding)

"""
# ANALYSIS
X_atoms_train_ = train_atoms_embedding.reshape([train_atoms_embedding.shape[0], 
                                        6 * 6])
X_bonds_train_ = train_bonds_embedding.reshape([train_bonds_embedding.shape[0], 
                                        6 * 6])

X_atoms_test_ = gen_atoms_embedding.reshape([gen_atoms_embedding.shape[0],
                                      6 * 6])
X_bonds_test_ = gen_bonds_embedding.reshape([gen_bonds_embedding.shape[0], 
                                      6 * 6])

pca_1 = PCA(n_components = 2)
X_atoms_train_ = pca_1.fit_transform(X_atoms_train_)
X_atoms_test_ = pca_1.transform(X_atoms_test_)

pca_2 = PCA(n_components = 2)
X_bonds_train_ = pca_2.fit_transform(X_bonds_train_)
X_bonds_test_ = pca_2.transform(X_bonds_test_)

# Atoms Distribution
plt.close()
plt.scatter(X_atoms_train_[:,0], X_atoms_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("train_atom_dist.png")
#plt.close()
plt.scatter(X_atoms_test_[:,0], X_atoms_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("test_atom_dist.png")
####

# Bonds Distribution
plt.close()
plt.scatter(X_bonds_train_[:,0], X_bonds_train_[:,1], alpha = 0.3, c = 'blue');
plt.savefig("train_bonds_dist.png")
#plt.close()
plt.scatter(X_bonds_test_[:,0], X_bonds_test_[:,1], alpha = 0.3, c = 'red');
plt.savefig("test_bonds_dist.png")
# 31/500 failed (N = 10000)
# 2/50 failed (N = 50000)
"""
#output.reset_index(drop = True, inplace = True)
output2.reset_index(drop = True, inplace = True)
output2.to_csv('./../experiments/regular/trans1_20screen_NODUP.csv', index = False)
"""with open('gen_pickles.pickle', 'wb') as f:
    pickle.dump(gen_unique_pickles, f)
"""

explained_variance_R2_pred_des = explained_variance_score(output['des_cv'], output['pred_cv'])
print ("explained_varice_R2_pred_des", explained_variance_R2_pred_des)
rsquared = r2_score (output['des_cv'], output['pred_cv'])
print ("r squared r**2", rsquared)

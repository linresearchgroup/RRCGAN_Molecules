# Task:
# Test the accuracy of Regression on generated samples:

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
# loading SMILES data using Chainer Chemistry
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor, construct_atomic_number_array

from tensorflow.keras.models import Model, load_model
from rdkit import Chem


preprocessor = GGNNPreprocessor()
"""
data = get_molnet_dataset('qm9',
                          labels = 'cv',
                          preprocessor = preprocessor,
                          return_smiles = True,
                          frac_train = 1.0,
                          frac_valid = 0.0,
                          frac_test = 0.0
                         )

"""
# change the position of pd dataframe based on
# heading of the .csv file. 
#data_gen = pd.read_csv('./../experiments/regular_9HA_6b6latent/Regular_noscreen.csv')
data_gen = pd.read_csv ('./Regular_validate_regressor.csv')
gen_smiles, dft_cv_ = data_gen.iloc[:, 0].values, data_gen.iloc[:, 3].values

print ('First SMILES in the file: ', gen_smiles[0])
print ('First pred. Cv during generation: ', dft_cv_[0])
print ('# of initial samples: ', len(dft_cv_))

# Normalize heat capacity using same min and max for generation
s_min_norm = 20.939
s_max_norm = 42.237
dft_cv = (dft_cv_ - s_min_norm) / (s_max_norm - s_min_norm)

with open('./../data/trainingsets/60000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
tokenizer [0] = ' '
print (tokenizer)

with open ('./../data/trainingsets/Data.pickle', 'rb') as f: 
    data = pickle.load(f)

# get the key from the value
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

X_smiles = []
X_gen_smiles = []

X_atoms = []
X_gen_atoms = []

X_bonds = []
X_gen_bonds = []
y = []

atom_lengths = []
atom_max = []
bonds_lengths = []


for smiles in data['smiles'][0]:
    smiles += '.'
    X_smiles.append(smiles)
X_smiles = X_smiles[1000:50000]
print ('qm9 shape', len(X_smiles))

for smile in gen_smiles:
    smile += '.'
    X_gen_smiles.append(smile)

for d in data['dataset'][0]:

    atom_lengths.append(len(d[0]))
    atom_max.append(np.max(d[0]))
    bonds_lengths.append(d[1].shape[1])
    y.append(d[2])

# subsample the qm9
y = y[1000:50000]
print ('heat capacity #', len(y))
X_atoms = []
X_bonds = []


y_norm = [(n-s_min_norm )/(s_max_norm - s_min_norm) for n in y]

# converting texts smiles to numbers
MAX_NB_WORDS = 23
MAX_SEQUENCE_LENGTH = 35

X_smiles_ = []
for smiles in X_smiles:
    c_smiles = []
    for c in smiles:
        c_smiles.append(get_keys_from_value(d=tokenizer, val=c)[0])    
    X_smiles_.append(c_smiles)
X_smiles = X_smiles_
X_gen_smiles_ = []
for smiles in X_gen_smiles:
    c_smiles = []
    for c in smiles:
        c_smiles.append(get_keys_from_value(d=tokenizer, val=c)[0])  
    X_gen_smiles_.append(c_smiles)
X_gen_smiles = X_gen_smiles_

X_smiles = pad_sequences(X_smiles,
                         maxlen = MAX_SEQUENCE_LENGTH,
                         padding = 'post')
X_gen_smiles = pad_sequences(X_gen_smiles,
                             maxlen = MAX_SEQUENCE_LENGTH,
                             padding = 'post')
X_smiles = to_categorical(X_smiles, num_classes = 23)
X_gen_smiles = to_categorical(X_gen_smiles, num_classes = 23)

print ('X_smiles shape', X_smiles.shape)
print ('X gen smiles shape', X_gen_smiles.shape)
SHAPE = list(X_smiles.shape) + [1]
X_smiles = X_smiles.reshape(SHAPE)
print ("X_smiles shape: ", X_smiles.shape)
SHAPE = list(X_gen_smiles.shape) + [1]
X_gen_smiles = X_gen_smiles.reshape(SHAPE)
print ("X_gen_smiles shape:  ", X_gen_smiles.shape)

y = np.asarray(y).reshape([-1])
dft_cv = np.asarray(dft_cv).reshape([-1])
print ('dft cv shape', dft_cv.shape)
encoder = load_model('./../data/nns_9HA_noemb_6b6/encoder_newencinp.h5')
decoder = load_model('./../data/nns_9HA_noemb_6b6/decoder_newencinp.h5')

regressor = load_model('./../data/nns_9HA_noemb_6b6/regressor.h5')
regressor_top = load_model('./../data/nns_9HA_noemb_6b6/regressor_top.h5')
print (".h5 was read")
print ('first gen smiles: ', X_gen_smiles[0])
print ('first qm9 smiles: ', X_smiles[0])
atoms_embedding, bonds_embedding, _ = encoder.predict([X_gen_smiles])
atoms_embedding_qm9, bonds_embedding_qm9, _ = encoder.predict([X_smiles])
# Validating the regressor
#====#
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for train data: {}'.format(r2_score(dft_cv, pred.reshape([-1]))))
print ("prediction on train: ", pred[1:10])
print ("true train: ", dft_cv[1:10])
mse_test_normalized = mean_squared_error(dft_cv, pred.reshape([-1]))
mse_test = mse_test_normalized * ((s_max_norm - s_min_norm)**2)
print ('norm. test MSE: ', mse_test_normalized, 'test MSE: ', mse_test)
print ('norm. test RMSE: ', mse_test_normalized**0.5, 'test RMSE: ', mse_test**0.5)
output = {}
print (gen_smiles[0])
print (pred[0])
print (dft_cv[0])
 
output['SMILES'] = list(gen_smiles)
output['pred_cv'] = list(pred)
output['DFT_cv'] = list(dft_cv)
output = pd.DataFrame(output)
output.reset_index(drop=True, inplace = True)
output.to_csv ('./Regular_regressortested_predictedvalue_DFT.csv', index=False)

pred = regressor.predict([atoms_embedding_qm9, bonds_embedding_qm9])
print('Current R2 on Regressor for train data: {}'.format(r2_score(y_norm, pred.reshape([-1]))))
print ("prediction on train: ", pred[1:10])
print ("true train: ", y_norm[1:10])



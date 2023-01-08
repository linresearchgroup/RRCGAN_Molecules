# Task:
# Turn data into "images"
# Two networks
# GAN: generate atoms and bonds (adjacency) layers
# simple CNN: turning SMILES layer to atoms+bonds layers

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

# loading SMILES data using Chainer Chemistry
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor, construct_atomic_number_array


from rdkit import Chem

"""Chem.MolFromSmiles('CC1CC(O)C2(CC2)O1')"""

preprocessor = GGNNPreprocessor()
#atom_num = construct_atomic_number_array()
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
data_gen = pd.read_csv('./Higher42_CV_DFT_HA10orless.csv')
gen_smiles, DFT_cv = data_gen.iloc[:,0].values, data_gen.iloc[:,1].values

print (gen_smiles[0])
print (DFT_cv[0])

X_gen_smiles = []
X_smiles = []

print (len(DFT_cv))


for smile in gen_smiles:
    smile += '.'
    X_gen_smiles.append(smile)

with open ('./../data/trainingsets/Data.pickle', 'rb') as f:
    data = pickle.load(f)

for smiles in data['smiles'][0]:
    smiles += '.'
    X_smiles.append(smiles)

X_gen_atoms = []
atom_lengths = []
atom_max = []
bonds_lengths = []
# get the mol from rdkit and then get atomic number of atoms from chainer_chemistry
for smile in gen_smiles:
    mol = Chem.MolFromSmiles(smile)
    X_gen_atoms_ = construct_atomic_number_array(mol)
    X_gen_atoms.append(X_gen_atoms_)


# excluding long atoms to use the same model trained on qm9 data
exclude = []
max_gen_atoms = 11
for i,ii in enumerate(X_gen_atoms):
    if len(ii) > max_gen_atoms:
        exclude.append(i)
print (exclude)
exclude = np.asarray(exclude, dtype = 'int')
idx = np.setdiff1d(np.arange(len(DFT_cv)), exclude)
idx = np.asarray(idx, dtype = 'int')
X_gen_atoms_ = []
X_gen_smiles_ = []
gen_smiles_ = []
pred_cv_ = []
DFT_cv_ = []
for i,ii in enumerate (idx):
    X_gen_atoms_.append (X_gen_atoms [ii])
    gen_smiles_.append (gen_smiles [ii])
    X_gen_smiles_.append(X_gen_smiles[ii])
    DFT_cv_.append  (DFT_cv [ii])

X_gen_atoms = X_gen_atoms_
gen_smiles = gen_smiles_
DFT_cv = DFT_cv_
X_gen_smiles = X_gen_smiles_
print (len(DFT_cv))
print (len(X_gen_smiles))
print (len(gen_smiles))

"""
with open('database_SMILES.pickle', 'wb') as f:
    pickle.dump((X_smiles, X_atoms, X_bonds, y), f)

with open('database_gen_SMILES.pickle', 'wb') as f:
    pickle.dump((X_gen_smiles, X_gen_atoms, X_gen_bonds, y), f)
"""

MAX_NB_WORDS = 23
MAX_SEQUENCE_LENGTH = 35

tokenizer = Tokenizer(num_words = MAX_NB_WORDS,
                      char_level = True,
                      filters = '',
                      lower = False)
tokenizer.fit_on_texts(X_smiles)
#tokenizer.fit_on_texts(X_gen_smiles)

X_smiles = tokenizer.texts_to_sequences(X_smiles)
X_gen_smiles = tokenizer.texts_to_sequences(X_gen_smiles)

X_smiles = pad_sequences(X_smiles,
                         maxlen = MAX_SEQUENCE_LENGTH,
                         padding = 'post')
X_gen_smiles = pad_sequences(X_gen_smiles,
                             maxlen = MAX_SEQUENCE_LENGTH,
                             padding = 'post')

X_smiles = to_categorical(X_smiles)
X_gen_smiles = to_categorical(X_gen_smiles, num_classes = 23)

# TRAIN/VAL split gen_data
idx = np.random.choice(len(DFT_cv), int(len(DFT_cv) * 0.15), replace = False)
train_idx = np.setdiff1d(np.arange(len(DFT_cv)), idx)

X_gen_smiles, gen_smiles, DFT_cv = np.asarray(X_gen_smiles), np.asarray(gen_smiles), np.asarray(DFT_cv)

X_gen_smiles_test, gen_smiles_test, DFT_cv_test = X_gen_smiles[idx], gen_smiles[idx], DFT_cv[idx]
X_gen_smiles_train, gen_smiles_train, DFT_cv_train = X_gen_smiles[train_idx], gen_smiles[train_idx], DFT_cv[train_idx]


with open('./../data/trainingsets/60000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
tokenizer[0] = ' '

print (tokenizer)
print (X_gen_smiles[0])
print (gen_smiles[0])
print (DFT_cv[0])
print (max(DFT_cv))

output = {}
output['SMILES'] = gen_smiles
output['DFT_cv'] = DFT_cv
output = pd.DataFrame(output)
output.reset_index(drop = True, inplace = True)
output.to_csv ('./Higher42_CV_DFT_11HA.csv', index=False)

with open('./../data/trainingsets/highcv_DFT_trans1/highcv_DFT.pickle', 'wb') as f:
    pickle.dump((X_gen_smiles, gen_smiles, DFT_cv),f)

with open('./../data/trainingsets/highcv_DFT_trans1/highcv_DFT_train.pickle', 'wb') as f:
    pickle.dump((X_gen_smiles_train, gen_smiles_train, DFT_cv_train),f)

with open('./../data/trainingsets/highcv_DFT_trans1/highcv_DFT_test.pickle', 'wb') as f:
    pickle.dump((X_gen_smiles_test, gen_smiles_test, DFT_cv_test),f)



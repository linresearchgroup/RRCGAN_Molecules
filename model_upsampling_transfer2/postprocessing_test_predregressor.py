# Task:
# Test the accuracy of Regression on generated samples:
# First, prediction from the fake latent vectors (from Generator) by Regressor
# Second, prediction from getting atom and bond matrices of gen. smiles using Decoder

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

from sklearn.metrics import r2_score
# loading SMILES data using Chainer Chemistry
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor, construct_atomic_number_array

from tensorflow.keras.models import Model, load_model
from rdkit import Chem

"""Chem.MolFromSmiles('CC1CC(O)C2(CC2)O1')"""

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
# file headings: SMILES,des_cv,pred_cv,jobacks
# named pred_cv during generation as latent_pred
data_gen = pd.read_csv('./practive_postprocess.csv')
gen_smiles, latent_pred = data_gen.iloc[:,0].values, data_gen.iloc[:,2].values

print ('First SMILES in the file: ', gen_smiles[0])
print ('First pred. Cv during generation: ', latent_pred[0])
print ('# of initial samples: ', len(latent_pred))

# Normalize heat capacity using same min and max for generation
min_heat = 21.02
max_heat = 42.302
latent_pred = (latent_pred - min_heat) / (max_heat - min_heat)

with open ('./../data/trainingsets/Data.pickle', 'rb') as f: 
    data = pickle.load(f)

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

# get Mol from rdkit
# get atomic number of atoms from chainer_chemistry
for smile in gen_smiles:
    mol = Chem.MolFromSmiles(smile)
    X_gen_atoms_ = construct_atomic_number_array(mol)
    X_gen_atoms.append(X_gen_atoms_)
    X_gen_bonds_ = preprocessor.get_input_features(mol)[1]
    X_gen_bonds.append(X_gen_bonds_)

# excluding  >12HA  to use the same model trained on qm9 data
exclude = []
max_gen_atoms = 12
for i,ii in enumerate(X_gen_atoms):
    if len(ii) > max_gen_atoms:
        exclude.append(i)
print (exclude)
exclude = np.asarray(exclude, dtype = 'int')
idx = np.setdiff1d(np.arange(len(latent_pred)), exclude)
idx = np.asarray(idx, dtype = 'int')

X_gen_atoms_ = []
X_gen_bonds_ = []
gen_smiles_ = []
latent_pred_ = []
for i,ii in enumerate (idx):
    X_gen_atoms_.append (X_gen_atoms [ii])
    X_gen_bonds_.append (X_gen_bonds [ii])
    gen_smiles_.append (gen_smiles [ii])
    latent_pred_.append  (latent_pred [ii])

X_gen_atoms = X_gen_atoms_
X_gen_bonds =  X_gen_bonds_
gen_smiles = gen_smiles_
latent_pred = latent_pred_

print ("# of samples after excluding >12 HA: ", len(latent_pred))


for smiles in data['smiles'][0]:
    smiles += '.'
    X_smiles.append(smiles)
X_smiles = X_smiles[0:1000]
print ('qm9 shape', len(X_smiles))

for smile in gen_smiles:
    smile += '.'
    X_gen_smiles.append(smiles)

for d in data['dataset'][0]:

    atom_lengths.append(len(d[0]))
    atom_max.append(np.max(d[0]))
    bonds_lengths.append(d[1].shape[1])
    y.append(d[2])

# subsample the qm9
y = y[0:1000]
print ('heat capacity #', len(y))
X_atoms = []
X_bonds = []

# get Mol from rdkit
# get atomic number of atoms from chainer_chemistry
for smile in X_smiles:
    mol = Chem.MolFromSmiles(smile[:-1])
    X_atoms_ = construct_atomic_number_array(mol)
    X_atoms.append(X_atoms_)
    X_bonds_ = preprocessor.get_input_features(mol)[1]
    X_bonds.append(X_bonds_)

y_norm = [(n-min_heat )/(max_heat - min_heat) for n in y]

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

X_smiles = to_categorical(X_smiles, num_classes = 23)
X_gen_smiles = to_categorical(X_gen_smiles, num_classes = 23)

atom_max = np.max(atom_max) + 3
bonds_max = np.max(bonds_lengths) + 3

X_atoms_ = []
for atom in X_atoms:
    if len(atom) < atom_max:
        pad_len = atom_max - len(atom)
        atom = np.pad(atom, (0, pad_len))

    X_atoms_.append(atom)

X_atoms = np.asarray(X_atoms_)

print (X_atoms.shape)
X_atoms = to_categorical(X_atoms, num_classes=10)
print (X_atoms.shape)

X_gen_atoms_ = []
atom_max = atom_max
for atom in X_gen_atoms:
    if len(atom) < max_gen_atoms:
        pad_len = max_gen_atoms - len(atom)
        atom = np.pad(atom, (0, pad_len))

    X_gen_atoms_.append(atom)

#X_gen_atoms = np.asarray(X_gen_atoms_)
X_gen_atoms = np.vstack (X_gen_atoms_)
print (X_gen_atoms)
print ((X_gen_atoms.shape))
X_gen_atoms = to_categorical(X_gen_atoms, num_classes=10)
print ("after to_categorical", X_gen_atoms.shape)
X_bonds_ = []
print ("max Bonds ", bonds_max)
for bond in X_bonds:
    if bond.shape[1] < bonds_max:
        pad_len = bonds_max - bond.shape[1]
        bond = np.pad(bond, ((0,0),(0,pad_len),(0,pad_len)))

    X_bonds_.append(bond)

X_bonds = np.asarray(X_bonds_)
print (X_bonds.shape)
X_gen_bonds_ = []
print (bonds_max )
bonds_max = bonds_max
for bond in X_gen_bonds:
    if bond.shape[1] < bonds_max:
        pad_len = bonds_max - bond.shape[1]
        bond = np.pad(bond, ((0,0),(0,pad_len),(0,pad_len)))

    X_gen_bonds_.append(bond)

X_gen_bonds = np.asarray(X_gen_bonds_)
print (type(X_gen_bonds))
print (X_gen_bonds.shape)
SHAPE = list(X_smiles.shape) + [1]
X_smiles = X_smiles.reshape(SHAPE)
print ("X_smiles shape: ", X_smiles.shape)
SHAPE = list(X_gen_smiles.shape) + [1]
X_gen_smiles = X_gen_smiles.reshape(SHAPE)
print ("X_gen_smiles shape:  ", X_gen_smiles.shape)

y = np.asarray(y).reshape([-1])
print ("line 189, ", X_atoms.shape)
SHAPE = list(X_atoms.shape) + [1]
print (SHAPE)
X_atoms = X_atoms.reshape(SHAPE)
print (X_atoms.shape)
X_bonds = X_bonds.transpose([0,2,3,1])
print (X_bonds.shape)
SHAPE = list(X_gen_atoms.shape) + [1]
X_gen_atoms = X_gen_atoms.reshape(SHAPE)
print (X_gen_atoms.shape)
X_gen_bonds = X_gen_bonds.transpose([0,2,3,1])
print (X_gen_bonds.shape)
####
def norm(X: ndarray) -> ndarray:
    X = np.where(X == 0, -1.0, 1.0)
    return X

X_gen_atoms, X_gen_bonds = (norm(X_gen_atoms),
                                norm(X_gen_bonds))
X_atoms, X_bonds = (norm(X_atoms),
                            norm(X_bonds))

encoder = load_model('./../data/nns_12HA_noemb_6b6/encoder_new.h5')
decoder = load_model('./../data/nns_12HA_noemb_6b6/decoder_new.h5')

regressor = load_model('./../data/nns_12HA_noemb_6b6/regressor_new.h5')
regressor_top = load_model('./../data/nns_12HA_noemb_6b6/regressor_top_new.h5')
print (".h5 was read")

atoms_embedding, bonds_embedding, _ = encoder.predict([X_gen_atoms, X_gen_bonds])
atoms_embedding_qm9, bonds_embedding_qm9, _ = encoder.predict([X_atoms, X_bonds])
# Validating the regressor
#====#
pred = regressor.predict([atoms_embedding, bonds_embedding])
print('Current R2 on Regressor for train data: {}'.format(r2_score(latent_pred, pred.reshape([-1]))))
print ("prediction on train: ", pred[1:10])
print ("true train: ", latent_pred[1:10])

pred = regressor.predict([atoms_embedding_qm9, bonds_embedding_qm9])
print('Current R2 on Regressor for train data: {}'.format(r2_score(y_norm, pred.reshape([-1]))))
print ("prediction on train: ", pred[1:10])
print ("true train: ", y_norm[1:10])



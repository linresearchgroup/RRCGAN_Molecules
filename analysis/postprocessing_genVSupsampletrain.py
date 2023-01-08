# Task:
# postprocessing analysis
# use Joback method (using functional groups to calc. Cv)
# find the rep. with qm9 and save the final file in a .csv file
# Joback has 6% error on qm9 values
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

#! pip install rdkit-pypi
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
print ("!!!!!!!!!!!!!!!!!!!!!we are after importing rdkit!!!!!!!!!!!!!!!!!!")
#! pip install thermo
from thermo import Joback
# loading SMILES data using Chainer Chemistry
#! pip install chainer_chemistry
from chainer_chemistry.datasets.molnet import get_molnet_dataset
from chainer_chemistry.datasets.numpy_tuple_dataset import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import GGNNPreprocessor

# load the generated SMILES from the RCGAN Model
csv_name = './../experiments/regular_9HA_6b6latent/regularVStransfer_cvdist/Regular_NODUP_noscreen.csv'
gen_SMILES = pd.read_csv(csv_name)

gen_SMILES_initial = gen_SMILES
initial_num_samples = gen_SMILES.shape[0]
print ('Number of gen SMILES', gen_SMILES.shape[0])
sanitized_idx = []

# make sure about Canonical SMILES and sanitizing 
for i, s in enumerate (gen_SMILES['SMILES'].values):
    try:
         m = Chem.MolFromSmiles(s, sanitize=True)
         ss = Chem.MolToSmiles(m)
         gen_SMILES['SMILES'].values[i] = ss
         sanitized_idx.append(i)
    except:
         print (s, 'is not sanitized')
gen_SMILES = gen_SMILES.iloc[sanitized_idx]
gen_SMILES.reset_index(drop=True, inplace=True)

print ('Number of gen SMILES after getting Canonical', gen_SMILES.shape[0])
# plot the RE before excluding similar to qm9 samples
# plot the error bars <10%, <20%
re_less_10 = np.sum(gen_SMILES['Err_pred_des'].values <= 0.1)
print (
    'less than 10 % ', re_less_10,
    'from total ', gen_SMILES.shape[0],
    re_less_10/gen_SMILES.shape[0], "%")
re_less_20_big_10 = np.sum( (gen_SMILES['Err_pred_des'].values > 0.1) &  (gen_SMILES['Err_pred_des'].values <= 0.2) )
print (
    'less than 20 larger than 10% ', re_less_20_big_10,
    'from total ', gen_SMILES.shape[0],
    re_less_20_big_10/gen_SMILES.shape[0], "%")
re_big_20 = np.sum(gen_SMILES['Err_pred_des'].values > 0.2)
print ('larger than 20 % ', re_big_20, 'from total ', gen_SMILES.shape[0], re_big_20/gen_SMILES.shape[0], "%")
print ('total: ', re_less_10 + re_less_20_big_10 + re_big_20)

plt.figure(figsize = (10, 8))
frequencies = [re_less_10/len(gen_SMILES)*100, re_less_20_big_10/len(gen_SMILES)*100, re_big_20/len(gen_SMILES)*100]
freq_series = pd.Series(frequencies)
ax = freq_series.plot(kind='bar', color = ['darkgreen', 'orange', 'magenta'])
rects = ax.patches
x_labels = [ ]
labels = ['<10%', '10%~20%', '>20%']

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height + 3, label,fontsize = 25,
            ha='center', va='bottom')
ax.set_xticklabels(x_labels)
plt.ylim(0, 100)
plt.yticks(fontsize=25)
plt.ylabel("Percentage", fontsize=25)
plt.xlabel("Pred. vs. Des. RE", fontsize=25)
plt.savefig('Err_pred_des_dist_3part_noexcluding.png', dpi=300)

des_cv = np.array (gen_SMILES['des_cv'])
pred_cv = np.array (gen_SMILES['pred_cv'])
print ('r2 of des. vs. pred before removing similar to qm9: ', r2_score(des_cv, pred_cv))


# pick the samples with available Joback values
jobacks = []
validated_smiles = []
valid_ids = []
for count, s in enumerate(gen_SMILES['SMILES'].values):
    try:
        J = Joback(s)
        jobacks.append(J.Cpig(298.15) * 0.2390057361)
        validated_smiles.append(s)
        valid_ids.append(count)
    except:
        pass
print ("Total SMILES validated by Joback {} Vs. total gen_SMILES {}".\
        format(len(validated_smiles), len(gen_SMILES['SMILES'])))

gen_SMILES2 = gen_SMILES.iloc[valid_ids, :]
gen_SMILES2['jobacks'] = jobacks
print (gen_SMILES2.shape)

""" error using Joback method mean """
# Joback vs. Des.
gen_SMILES2['Err_joback_des'] = np.abs(
    (gen_SMILES2['des_cv'].values - gen_SMILES2['jobacks'].values)/
     gen_SMILES2['jobacks'].values)
print ("mean error Joback(gen_SMILES) Vs. des. value: ",
       np.mean(gen_SMILES2['Err_joback_des'].values))
# Joback vs. Pred.
gen_SMILES2['Err_joback_pred'] = np.abs(
    (gen_SMILES2['pred_cv'].values - gen_SMILES2['jobacks'].values)/
     gen_SMILES2['jobacks'].values)
print ("mean error Joback(gen_SMILES) Vs. pred. value: ",
       np.mean(gen_SMILES2['Err_joback_pred'].values))

gen_SMILES2.reset_index(drop = True, inplace = True)

csv_name = csv_name.replace('.csv', '')

""" screen candidates in generated SMILES with close to Joback values """
val_accurate = pd.DataFrame({'SMILES': [],
                             'des_cv': [],
                             'pred_cv': [],
                             'jobacks': []})
accurate = []

for i, s in enumerate (gen_SMILES2['SMILES'].values):
    if gen_SMILES2['Err_joback_des'].values[i] > 0:
        accurate.append(i)

for ii, a in enumerate (accurate):
    #print (" i and a from accurate",ii, a)
    val_accurate.loc[ii,:] = gen_SMILES2.iloc[a,:]

print ("the first smile in val_accurate: ",val_accurate['SMILES'].values[0])
for i, s in enumerate (val_accurate['SMILES'].values):
    #print (s)
    m = Chem.MolFromSmiles(s)
    ss = Chem.MolToSmiles(m)
    val_accurate['SMILES'].values[i] = ss

sort_val_accurate = val_accurate.sort_values ('des_cv')
#print (sort_val_accurate)

# accuracy of the the model Joback vs. predicted and desired Cv (accurate < 5%)
mean_err = np.mean(np.abs((val_accurate['pred_cv'].values -
                           val_accurate['jobacks'].values) /
                           val_accurate['jobacks'].values))
print ("mean error Joback(gen_SMILES) Vs.Predicted from regressor (for accurate Cv(<10%): ", mean_err)


mean_err = np.mean(np.abs((val_accurate['des_cv'].values -
                           val_accurate['jobacks'].values) /
                           val_accurate['jobacks'].values))
print ("mean error Joback(gen_SMILES) Vs.Desired from regressor (for accurate Cv(<10%): ", mean_err)


num_acc_l0p1 = np.sum(np.abs((gen_SMILES2['des_cv'].values -
                              gen_SMILES2['jobacks'].values) /
                              gen_SMILES2['jobacks'].values) < 0.1)

plt.scatter(gen_SMILES2['des_cv'].values, gen_SMILES2['jobacks'].values)
plt.savefig("Desired_VS_joback.png")

plt.clf()
plt.scatter(val_accurate['des_cv'].values, val_accurate['jobacks'].values)
plt.title("Accurate Desired Cv vs. Joback Cv")
plt.xlabel("Desired Cv")
plt.ylabel("Joback Cv")
plt.savefig("Desired_accurate_VS_joback.png")


plt.clf()
err_Desacc_job = ((val_accurate['des_cv'].values - val_accurate['jobacks'].values) / val_accurate['jobacks'].values)
err_Desacc_job = pd.Series(err_Desacc_job, name="err_Des_accurate_VS_Jobsck")
sns.distplot(err_Desacc_job)
plt.savefig("err_Des_accurate_VS_Jobsck.png")
plt.clf()
val_accurate.reset_index(drop = True, inplace = True)

#val_accurate.to_csv('gen_new_noscreen_all_joback.csv', index = False)
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


""" reading and preprocessing data"""

with open('./../data/trainingsets/60000_train_regular_qm9/train_GAN.pickle', 'rb') as f:
    X_smiles_gantrain, SMILES_gantrain, cv_gantrain = pickle.load(f)

with open('./../data/trainingsets/60000_train_regular_qm9/image_train.pickle', 'rb') as f:
    X_smiles_train, SMILES_train, X_atoms_train, X_bonds_train, y_train0 = pickle.load(f)

with open('./../data/trainingsets/60000_train_regular_qm9/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
tokenizer[0] = ' '


## Outlier removal 1.5*IQR rule
# Train samples
IQR = - np.quantile(y_train0, 0.25) + np.quantile(y_train0, 0.75)
lower_bound, upper_bound = np.quantile(y_train0, 0.25) - 1.5 * IQR, np.quantile(y_train0, 0.75) + 1.5 * IQR
idx = np.where((y_train0 >= lower_bound) & (y_train0 <= upper_bound))

y_train = y_train0[idx]
X_smiles_train = X_smiles_train[idx]
X_atoms_train = X_atoms_train[idx]
X_bonds_train = X_bonds_train[idx]


plt.clf()
sns.distplot(des_cv, color='red')
fig = sns.distplot(y_train, color='yellow')
fig = fig.get_figure()
fig.savefig("des_train_distributions.png") 

plt.clf()
sns.set_style("white")
mpl.rcParams['axes.linewidth'] = 3
fig, ax = plt.subplots(figsize = (15, 12))
mpl.rcParams['axes.linewidth'] = 3
ax.tick_params(axis='both', which='major', labelsize=35, width=5)
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel(r'C$\mathbf{_\nu}$', fontsize=35, fontweight="bold")
plt.ylabel('Density', fontsize=35, fontweight="bold")
sns.distplot(pred_cv, color='red')
fig = sns.distplot(y_train, color='blue')
ax.legend(['Generated', 'QM9'], fontsize=30, markerscale=100)
fig = fig.get_figure()
fig.savefig("pred_train_distributions.png", dpi=500, bbox_inches='tight')


print (pred_cv.shape)
print (cv_gantrain.shape)
print (np.min(des_cv))
print (np.max(des_cv))



with open('./../data/trainingsets/highcv_qm9/train_GAN.pickle', 'rb') as f:
    X_smiles_gantrain, SMILES_gantrain, _, __, cv_gantrain = pickle.load(f)

plt.clf()
sns.set_style("white")
mpl.rcParams['axes.linewidth'] = 3
fig, ax = plt.subplots(figsize = (15, 12))
mpl.rcParams['axes.linewidth'] = 3
ax.tick_params(axis='both', which='major', labelsize=35, width=5)
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel(r'C$\mathbf{_\nu}$', fontsize=35, fontweight="bold")
plt.ylabel('Density', fontsize=35, fontweight="bold")
#sns.distplot(pred_cv, color='red')
fig = sns.distplot(cv_gantrain, color='green')
ax.legend([r'C$\mathbf{_\nu}$> 42 QM9'], fontsize=30, markerscale=100)
fig = fig.get_figure()
fig.savefig("trans_train_distributions.png", dpi=500, bbox_inches='tight')

# read complete QM9
csv_name = './../experiments/regular_9HA_6b6latent/regularVStransfer_cvdist/qm9_cv.csv'
qm9all_SMILES = pd.read_csv(csv_name)
qm9all_cv = np.array(qm9all_SMILES['cv'])
# read trans iter. 1
csv_name = './../experiments/regular_9HA_6b6latent/regularVStransfer_cvdist/trans_NODUP_20screen.csv'
gen_SMILES = pd.read_csv(csv_name)
predtrans1_cv = np.array (gen_SMILES['pred_cv'])
# read trans iter. 0
csv_name = './../experiments/regular_9HA_6b6latent/regularVStransfer_cvdist/trans1_NODUP_20screen.csv'
gen_SMILES = pd.read_csv(csv_name)
predtrans1_cv = np.array (gen_SMILES['pred_cv'])

csv_name = './../experiments/regular_9HA_6b6latent/regularVStransfer_cvdist/trans2_NODUP_20screen.csv'
gen_SMILES = pd.read_csv(csv_name)
predtrans2_cv = np.array (gen_SMILES['pred_cv'])

# plotting qm9all, regular gen., iteration 1, and iteration 2
plt.clf()
sns.set_style("white")
mpl.rcParams['axes.linewidth'] = 6
fig, ax = plt.subplots(figsize = (15, 12))
mpl.rcParams['axes.linewidth'] = 6
ax.tick_params(axis='both', which='major', labelsize=35, width=5)
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel(r'C$\mathbf{_\nu}$', fontsize=35, fontweight="bold")
plt.ylabel('Density', fontsize=35, fontweight="bold")
fig = sns.distplot(qm9all_cv, color='blue', hist_kws={'alpha': 0.2})
sns.distplot(pred_cv, color='darkred')
sns.distplot(predtrans1_cv, color='green')
sns.distplot(predtrans2_cv, color='darkgreen')
ax.legend(['QM9', 'Regular Gen.', 'Iteration 1', 'Iteration 2'], 
            fontsize=30, markerscale=2)
fig = fig.get_figure()
fig.savefig("trans_pred_train_distributions.png", dpi=500, bbox_inches='tight')




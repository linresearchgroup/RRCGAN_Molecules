import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen as logp
from rdkit.Chem import rdMolDescriptors as tpsa
from rdkit.Chem import QED as QED

from chainer_chemistry.datasets.molnet import get_molnet_dataset
"""
data = get_molnet_dataset(
    'qm9', 
    labels = 'cv',
    return_smiles = True, 
    frac_train = 1.0,
    frac_valid = 0.0,
    frac_test = 0.0
)

with open ("Data.pickle", 'wb') as f:
    pickle.dump(data,f)
"""

with open ("Data.pickle", 'rb') as f:
   data = pickle.load(f)
SMILES = []
cv = []
cv_ = []
for smiles in data['smiles'][0]:
    SMILES.append(smiles)
    
for d in data['dataset'][0]:
    cv_.append(d[1])
cv = [d[0] for d in cv_]

print (cv[0])
print (type(cv))
#print (cv)

# put csvfile name here
csv_filename = 'Regular_NODUP_60ksam_13joback_DFT.csv'
gen_SMILES_data = pd.read_csv(csv_filename)

gen_SMILES = []
gen_cv = []
for s in gen_SMILES_data['SMILES'].values:
    gen_SMILES.append(s)

# if the file has DFT use that value, if not desired values
try:
    for h in gen_SMILES_data['DFT_cv'].values:
        gen_cv.append(h)
except:
    for h in gen_SMILES_data['des_cv'].values:
        gen_cv.append(h)


print (SMILES[1])
m = AllChem.MolFromSmiles(SMILES[1])
output = Chem.MolToMolBlock(m)
#print ("this is output: ", output)
#print ("this is re.sub", re.sub('[\W+\d+H]', '', SMILES[1]))

#print (len(re.sub('[\W+\d+H]', '', SMILES[1])))

print ("this is m: ", m)

coord = np.array([a.split()[:3] for a in output.split('\n')[4:(4+9)]]).astype(float)


print ("this is 2D coordinates", coord)
plt.scatter(coord[:,0], coord[:,1],
            c = ['black',
                 'blue',
                 'black',
                 'black',
                 'blue',
                 'black',
                 'black',
                 'red',
                 'blue'],
            s = [6 * 40,
                 7 * 40,
                 6 * 40,
                 6 * 40,
                 7 * 40,
                 6 * 40,
                 6 * 40,
                 8 * 40,
                 7 * 40],
            alpha = 0.5)
plt.savefig("smiles.png")
coords = []
pure_atoms = []
_3Ds = 0
gen_coords = []
gen_pure_atoms = []
gen_3Ds = 0

for smiles in SMILES:
    m = AllChem.MolFromSmiles(smiles)
    output = Chem.MolToMolBlock(m)
    if output.split('\n')[1].split()[1]=='3D':
        _3Ds += 1
    
    pure_atom = re.sub('[\W+\d+H]', '', smiles)
    stop = len(pure_atom)
    pure_atoms.append(pure_atom)
    
    coord = np.array([a.split()[:3] for a in output.split('\n')[4:(4+stop)]]).astype(float)
    coords.append(coord)

valid = []
for i,smiles in enumerate (gen_SMILES):
    try:
     m = AllChem.MolFromSmiles(smiles)
     output = Chem.MolToMolBlock(m)
     if output.split('\n')[1].split()[1]=='3D':
        gen_3Ds += 1

     pure_atom = re.sub('[\W+\d+H]', '', smiles)
     stop = len(pure_atom)
     gen_pure_atoms.append(pure_atom)

     coord = np.array([a.split()[:3] for a in output.split('\n')[4:(4+stop)]]).astype(float)
     gen_coords.append(coord)
     valid.append (i)
    except:
     pass
gen_SMILES = [gen_SMILES[i] for i in valid]
gen_cv = [gen_cv[i] for i in valid]

print ("this is _3Ds: ",_3Ds)
print ("len(coords) {}, len(SMILES) {}, len(cv) {}, len(pure_atoms) {}".format(len(coords), len(SMILES), len(cv), len(pure_atoms)))

print ("this is gen_3Ds: ",gen_3Ds)
print ("len(coords_gen) {}, len(gen_SMILES) {}, len(gen_cv) {}, len(gen_pure_atoms) {}".format(len(gen_coords), len(gen_SMILES), len(gen_cv), len(gen_pure_atoms)))
"""
print (coords[1])
with open('coordinates.pickle', 'wb') as f:
    pickle.dump((coords, SMILES, cv, pure_atoms), f)
"""
"""----"""

####

features = {
    'MolWt': Descriptors.MolWt,
    'HeavyAtomCount': Descriptors.HeavyAtomCount,
    'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt,
    'NumHAcceptors': Descriptors.NumHAcceptors,
    'NumHDonors': Descriptors.NumHDonors,
    'NumHeteroatoms': Descriptors.NumHeteroatoms,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumValenceElectrons': Descriptors.NumValenceElectrons,
    'NumAromaticRings': Descriptors.NumAromaticRings,
    'NumSaturatedRings': Descriptors.NumSaturatedRings,
    'NumAliphaticRings': Descriptors.NumAliphaticRings,
    'NumRadicalElectrons': Descriptors.NumRadicalElectrons,
    'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
    'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles,
    'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
    'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles,
    'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles,
    'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles, 
    'Logp': logp.MolLogP,
    'TPSA': tpsa.CalcTPSA,
    'QED' : QED.default
}

out_data = {}
for f in features.keys():
    out_data[f] = []

for i, smiles in enumerate(SMILES):
    
    if (i + 1) % 5000 == 0:
        print('Currently processed: {}/{}'.format(i+1, len(SMILES)))
    
    m = AllChem.MolFromSmiles(smiles)
    
    for k, v in features.items():
        out_data[k].append(v(m))

#cv = cv.replace
out_data['heat_capacity'] = cv

out_data = pd.DataFrame(out_data)

out_data.to_csv('Features_hc_qm9.csv', index = False)

gen_out_data = {}
for f in features.keys():
    gen_out_data[f] = []

for i, smiles in enumerate(gen_SMILES):

    if (i + 1) % 5000 == 0:
        print('Currently processed: {}/{}'.format(i+1, len(gen_SMILES)))

    m = AllChem.MolFromSmiles(smiles)
    try:
        for k, v in features.items():
            gen_out_data[k].append(v(m))
    except:
        pass

gen_out_data['heat_capacity'] = gen_cv

gen_out_data = pd.DataFrame(gen_out_data)

gen_out_data.to_csv('Features{}'.format(csv_filename), index = False)


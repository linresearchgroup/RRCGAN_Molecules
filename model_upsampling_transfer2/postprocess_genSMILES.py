import numpy as np
import pandas as pd


gen_smiles = pd.read_csv('final_data_Feat_Cv_smiles.csv')
gen_smiles1 = pd.read_csv('final_data_Feat_Cv_smiles1.csv')
rep = []
print (gen_smiles.shape)
print (gen_smiles1.shape)

smiles = gen_smiles.iloc[:,-1]
smiles1 = gen_smiles1.iloc[:,-1]

frames = [gen_smiles, gen_smiles1]
result = pd.concat (frames)
print (result)
print (result.shape)
gen_SMILES = result.drop_duplicates (subset = ['SMILES'])
print (gen_SMILES)
print (gen_SMILES.shape)

gen_SMILES.to_csv('gen_SMILES.csv')

# RRCGAN_publish_final
Title: 
De Novo Design of Molecules Towards Biased Properties via a Deep Generative Framework and Iterative Transfer Learning

De Novo design of molecules with targeted properties represents a new frontier in molecule development. Despite enormous progress, two main challenges remain, i.e., (i) generation of novel molecules with targeted and quantifiable properties; (ii) generated molecules having property values beyond the range in the training dataset. To tackle these challenges, we propose a novel reinforced regressional and conditional generative adversarial network (RRCGAN) to generate chemically valid, drug-like molecules with targeted heat capacity (Cv) values as a proof-of-concept study. As validated by DFT, ~80% of the generated samples have a relative error (RE) of < 20% of the targeted Cv values. To bias the generation of molecules with the Cv values beyond the range of the original training molecules, transfer learning was applied to iteratively retrain the RRCGAN model. After only two iterations of transfer learning, the mean Cv of the generated molecules increases to 44.0 cal/(mol·K) from the mean value of 31.6 cal/(mol·K) shown in the initial training dataset. This demonstrated computation methodology paves a new avenue to discovering drug-like molecules with biased properties, which can be straightforwardly repurposed for optimizing individual or multi-objective properties of various matters. 

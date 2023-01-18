For demo purposes, we put a complete set of codes and datasets in Demo folder. 
We reduced the number of training samples in order to keep the size under 25MB for uploading in Github. 
For the real experiment, we train the model on ~60,000 training samples (close to half of original QM9 samples). 

# Running "preprocesses_version_0_3_demo.py" 
will generate the necessary training and testing data by downloading QM9 library and sampling from it.
Running that file generates "./image_train_demo.pickle", "./image_test_demo.pickle", and "./tokenizer.pickle"

### Output
"./image_train_demo.pickle", "./image_test_demo.pickle": 
include information about the samples (their SMILES strings, one-hot encoded, bond and atom information, and Cv values).
 
"./tokenizer.pickle" 
includes the mapping from SMILES strings to their one-hot encoded. We save it to use the same mapping through the entire process. 

# Running "embedding_version_0_3_60ksam_encodernewinput.py" 
After generating required data sets, one needs to traing AE including the encoder and decoder.
For demo, we chose 10 epochs, but the real one needs ~800 epochs. Running that file generates three files keeping the encoder, decoder, and ae (the combined model) weights. 
### Output:
"encoder_newenc.h5", "decoder_newenc.h5", and "ae_model_newenc.h5".
It also generate some *.png images of one-hot encoded SMLILES. 
The code also generates some one-encoded SMILES to compare the converted one-hot encoded and the input one. 
Ideally, they should be the same. For demo, we only run it for 10 epochs. Also, only ~3000 samples were used for training. 

# Running "main_version_0_5_training_normaltrain.py"
After training and saving the Encoder and Decoder, one needs to run the main model, named "main_version_0_5_training_normaltrain.py". First, it uses the saved encoder and decoder to train the regressor. The trained regressor then is used inside the GAN model to generate molecules with targeted (desired) properties. 
When it is running, the code print the following:
### Printing
#### 1) Current epoch: 
shows the epochs count for training the overal GAN model using the trained AE and Regressor. 
#### 2) D Loss Real: 
Discriminator loss for detecting real samples. 
#### 3) D Loss Fake: 
Discriminator loss for detecting fake samples. 
#### 4) D Loss: 
The average of the above that is considered discriminator loss. 
#### 5) G Loss: 
Generator loss.
#### 6) R Loss: 
Regressor loss. 
#### 7) Currently valid SMILES (No chemical_beauty and sanitize off): 
valid samples out of 1000 that is generated in reinforcement center. 
#### 9) Currently valid SMILES Unique (No chemical_beauty and sanitize off): 
valid and unique samples out of 1000. If the model training traps in mode collapse, there are many repetitive samples--> the number of unique and valid is far fewer than the number of valid. 
#### 10) Currently valid SMILES Sanitized: 
valid and chemically sanitized samples out of 1000
#### 11) Currently valid Unique SMILES Sanitized: 
Unique, valid, and chemically sanitized samples out of 1000
#### 13) Currently satisfying SMILES: 
Valid, chemically sanitized, and within 20% error samples out of 1000. We compare the predicted value from Regressor and targeted value to calculate the accuracy. 
#### 14) Currently unique satisfying generation: 
Unique, valid, chemically sanitized, and within 20% error samples out of 1000. We compare the predicted value from Regressor and targeted value to calculate the accuracy.
### Output
Finally, the main model generate the final samples within the targeted range. It will print the accuracy of the model on generated data comparing the targeted and predicted values. It will outputs "demo.csv" and "demo_NODUP.csv" files with SMILES strings, their targeted and predicted heat capacity values, and their error in a csv format. 


# To go from demo to real running, one needs to increase the training samples to around 60K samples and run the model on more epochs to reach the desired accuracies. 

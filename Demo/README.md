For demo purposes, we put a complete set of codes and datasets in Demo folder. 
We reduced the number of training samples in order to keep the size under 25MB for uploading in Github. 
For the real experiment, we train the model on ~60,000 training samples (close to half of original QM9 samples). 

1) Running "preprocesses_version_0_3_demo.py" will generate the necessary training and testing data by downloading QM9 library and sampling from it.
Running that file generates "./image_train_demo.pickle", "./image_test_demo.pickle", and "./tokenizer.pickle"

"./image_train_demo.pickle" and "./image_test_demo.pickle": include information about the samples (their SMILES strings, one-hot encoded, bond and atom information, and Cv values).
"./tokenizer.pickle" includes the mapping from SMILES strings to their one-hot encoded. We save it to use the same mapping through the entire process. 

2) After generating necessary dataset, one needs to run "embedding_version_0_3_60ksam_encodernewinput.py" to traing AE including the encoder and decoder.
For demo, we chose 10 epochs, but the real one needs ~800 epochs. Running that file generates three files keeping the encoder, decoder, and ae (the combined model) weights. 
Output: "encoder_newenc.h5", "decoder_newenc.h5", and "ae_model_newenc.h5".
The code also generates some one-encoded SMILES to compare the converted one-hot encoded and the input one. 
Ideally, they should be the same. For demo, we only run it for 10 epochs and only ~10,000 samples. 

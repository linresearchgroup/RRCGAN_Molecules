# RRCGAN 
Submitted to Nature Computational Science.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

RRCGAN is a deep generative model using a Generative Adversarial Network (GAN) cmobined with a Regressor to generate molecules with targeted properties. It is puerly run in Python. Using GPU is necessary, otherwise the running takes a lot!

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Setting up the environment](#setting-up-the-development-environment)
- [License](#license)

# Overview
``RRCGAN`` is a generative GAN model designed to generate small molecules with targeted properties. ``RRCGAN``, a generative deep learning model, has been built in Keras from Tensorflow that can easily installed on personal computers. Having a GPU is recommended to accelerate each epochs of learning. The packages used in RRCGAN can be installed on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows).


# System Requirements
## Hardware requirements
`RRCGAN` requires only a standard computer with GPU and enough RAM. 

## Software requirements
### Python Dependencies
`RRCGAN` mainly depends on the Python scientific stack, Keras form Tensorflow, and RDKit.

```
numpy
scipy
scikit-learn
pandas
seaborn
sklearn
tensorflow
matplotlib
chainer chemistry
RDKit
```

# Installation Guide:
The only challenge for running the model is to set up the Tensorflow-gpu. One should follow specific version of Tensorflow and Nvidia drivers to make it work. The necessary packages and the built conda environment used is `environment.yml`.
We primarily used Lewis Cluster from University of Missouri-Columbia for running the code. The following is the information of a personal machine that was used for running the code.
GPU Nvidia RTX 2080 Super, Cuda version: 10.1, cuDNN: 7.6, Tensorflow: 2.11.0


# License
This project is covered under the **Apache 2.0 License**.

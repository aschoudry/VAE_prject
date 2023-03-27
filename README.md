# Variational Autoencoder for Genome Data Compression
This repository contains an implementation of a Variational Autoencoder (VAE) to compress genome data and obtain a latent representation for downstream tasks such as cancer classification and progression analysis.
# Dataset
The dataset used in this project consists of genomic sequences of patients diagnosed with cancer. Each sample in the dataset is represented by a vector of features corresponding to various genomic regions. The data is preprocessed and normalized before being fed into the VAE.

# Model Architecture
The VAE architecture used in this project consists of an encoder and a decoder network. The encoder network takes in the preprocessed input data and outputs a mean and log variance of the latent distribution. The decoder network takes in a sample from the latent distribution and reconstructs the input data.

# Training
The model is trained using stochastic gradient descent with a learning rate of 0.001 and a batch size of 64. The loss function is a combination of the reconstruction error and the KL divergence of the latent distribution from a Gaussian prior. The model is trained for 100 epochs and the best performing model is saved.

# Evaluation
The trained VAE is evaluated on downstream tasks such as cancer classification and progression analysis. The compressed latent representation obtained from the VAE is used as input for these tasks. The performance of the VAE is compared to other compression methods and evaluated using appropriate metrics.

# Requirements
  Python 3.7
  PyTorch
  Pandas
  NumPy

# Usage
To train the VAE, run the train.py script with the appropriate arguments. The trained model will be saved in the saved_models directory. To evaluate the VAE, run the evaluate.py script with the appropriate arguments.

# References
[1] Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

[2] Doersch, C. (2016). Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.






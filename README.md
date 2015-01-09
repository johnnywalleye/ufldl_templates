## Python Templates for Stanford Unsupervised Feature Learning and Deep Learning Tutorial

Tutorial Website: http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial

This repo contains python templates for the exercises, based on the completed exercises here: https://github.com/jatinshah/ufldl_tutorial

### Exercise 1: Sparse Autoencoder

http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder

Sparse Autoencoder implementation, learning/visualizing features on sample data

Data for this exercise is in this repo at `data/IMAGES.mat`

Files to update:
* [gradient.py](gradient.py): Functions to compute & check cost and gradient
* [sparse_autoencoder.py](sparse_autoencoder.py): Sparse autoencoder cost & gradient functions

Other files used:
* [sample_images.py](sample_images.py): Load sample images for testing sparse auto-encoder
* [display_network.py](display_network.py): Display visualized features
* [train.py](train.py): Train sparse autoencoder with MNIST data and visualize learnt featured

To run this exercise, start python in the home directory for this repo and run:
`execfile('ex_1_sparse_autoencoder.py')`

### Exercise 2: Sparse Autoencoder, Vectorized

http://ufldl.stanford.edu/wiki/index.php/Exercise:Vectorization

Sparse Autoencoder vectorized implementation, learning/visualizing features on MNIST data

Data for this exercise is in this repo at `data/mnist/train-images-idx3-ubyte`

Files to update:
* [sparse_autoencoder.py](sparse_autoencoder.py): Sparse autoencoder cost & gradient functions

Other files used:
* [display_network.py](display_network.py): Display visualized features
* [load_MNIST.py](load_MNIST.py): Load MNIST images
* [train.py](train.py): Train sparse autoencoder with MNIST data and visualize learnt featured

To run this exercise, start python in the home directory for this repo and run:
`execfile('ex_2_sparse_autoencoder_vect.py')`

### Exercise 3a: PCA in 2D

http://deeplearning.stanford.edu/wiki/index.php/Exercise:PCA_in_2D

Implement PCA, PCA whitening & ZCA whitening

Data for this exercise is in this repo at `data/pcaData.txt`

Files to update:
* [ex_3a_pca_in_2d.py](ex_3a_pca_in_2d.py): PCA in 2d implementation

Other files used:
None

To run this exercise, start python in the home directory for this repo and run:
`execfile('ex_3a_pca_in_2d.py')`

### Exercise 3a: PCA and Whitening

http://deeplearning.stanford.edu/wiki/index.php/Exercise:PCA_and_Whitening

Implement PCA, PCA whitening & ZCA whitening

Data for this exercise is in this repo at `data/IMAGES_RAW.mat`

Files to update:
* [ex_3b_pca_gen.py](ex_3b_pca_gen.py): PCA and Whitening implementation

Other files used:
* [display_network.py](display_network.py): Display visualized features
* [sample_images.py](sample_images.py): Load sample images for testing PCA and Whitening

To run this exercise, start python in the home directory for this repo and run:
`execfile('ex_3b_pca_gen.py')`
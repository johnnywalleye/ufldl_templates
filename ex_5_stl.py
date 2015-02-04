import load_MNIST
import numpy as np
import sparse_autoencoder
import scipy.optimize
import display_network
import softmax

## ======================================================================
#  STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to
#  change the parameters below.

input_size = 28 * 28
num_labels = 5
hidden_size = 196

sparsity_param = 0.1  # desired average activation of the hidden units.
lambda_ = 3e-3  # weight decay parameter
beta = 3  # weight of sparsity penalty term

## ======================================================================
#  STEP 1: Load data from the MNIST database
#
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
labels = load_MNIST.load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')

unlabeled_index = np.argwhere(labels >= 5).flatten()
labeled_index = np.argwhere(labels < 5).flatten()

num_train = round(labeled_index.shape[0] / 2)
train_index = labeled_index[0:num_train]
test_index = labeled_index[num_train:]

unlabeled_data = images[:, unlabeled_index]

train_data = images[:, train_index]
train_labels = labels[train_index]

test_data = images[:, test_index]
test_labels = labels[test_index]

print '# examples in unlabeled set: {0:d}\n'.format(unlabeled_data.shape[1])
print '# examples in supervised training set: {0:d}\n'.format(train_data.shape[1])
print '# examples in supervised testing set: {0:d}\n'.format(test_data.shape[1])

## ======================================================================
#  STEP 2: Train the sparse autoencoder
#  This trains the sparse autoencoder on the unlabeled training
#  images.

#  Randomly initialize the parameters
theta = sparse_autoencoder.initialize(hidden_size, input_size)

## ----------------- YOUR CODE HERE ----------------------
#  Find opt_theta by running the sparse autoencoder on
#  unlabeledTrainingImages
opt_theta = theta


# Visualize the weights
W1 = opt_theta[0:hidden_size * input_size].reshape(hidden_size, input_size).transpose()
display_network.display_network(W1)

##======================================================================
## STEP 3: Extract Features from the Supervised Dataset
#
#  You need to complete the code in feedForwardAutoencoder.m so that the
#  following command will extract features from the data.

train_features = sparse_autoencoder.sparse_autoencoder(opt_theta, hidden_size,
                                                       input_size, train_data)

test_features = sparse_autoencoder.sparse_autoencoder(opt_theta, hidden_size,
                                                      input_size, test_data)

##======================================================================
## STEP 4: Train the softmax classifier

lambda_ = 1e-4
softmax = None
## ----------------- YOUR CODE HERE ----------------------
#  Use softmaxTrain.m from the previous exercise to train a multi-class
#  classifier. 

#  Use lambda = 1e-4 for the weight regularization for softmax

# You need to compute softmaxModel using softmaxTrain on trainFeatures and
# trainLabels

##======================================================================
## STEP 5: Testing

predictions = softmax.softmax_predict((opt_theta, input_size, num_classes), test_features)
print "Accuracy: {0:.2f}%".format(100 * np.sum(predictions == test_labels, dtype=np.float64) / test_labels.shape[0])

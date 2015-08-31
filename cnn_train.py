import cnn_cost
import cnn_init_params
import gradient
import load_MNIST
import min_func_sgd
import numpy as np

## Convolution Neural Network Exercise

#  Instructions
#  ------------
#
#  This file contains code that helps you get started in building a single.
#  layer convolutional nerual network. In this exercise, you will only
#  need to modify cnnCost.m and cnnminFuncSGD.m. You will not need to
#  modify this file.

##======================================================================
## STEP 0: Initialize Parameters and Load Data
#  Here we initialize some parameters used for the exercise.

# Configuration
image_dim = 28
num_classes = 10  # Number of classes (MNIST images fall into 10 classes)
filter_dim = 9  # Filter size for conv layer
num_filters = 20  # Number of filters for conv layer
pool_dim = 2  # Pooling dimension, (should divide image_dim-filter_dim+1)

# Load MNIST Train
images = load_MNIST.load_MNIST_images('data/mnist/train-images-idx3-ubyte')
images = images.reshape([image_dim, image_dim, images.shape[1]])
images = images[:, :, :100]
labels = load_MNIST.load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')
labels[labels == 0] = 10  # Remap 0 to 10
labels = labels[:100]

# Initialize Parameters
theta = cnn_init_params.cnn_init_params(
    image_dim, filter_dim, num_filters, pool_dim, num_classes)

##======================================================================
## STEP 1: Implement convNet Objective
#  Implement the function cnn_cost()
cost, grad, _ = cnn_cost.cnn_cost(theta, images, labels, num_classes, filter_dim,
                                      num_filters, pool_dim)

##======================================================================
## STEP 2: Gradient Check
#  Use the file gradient.py to check the gradient
#  calculation for your cnnCost.m function.  You may need to add the
#  appropriate path or copy the file to this directory.

DEBUG = False  # set this to true to check gradient
if DEBUG:
    # To speed up gradient checking, we will use a reduced network and
    # a debugging data set
    db_numFilters = 2
    db_filterDim = 9
    db_poolDim = 5
    db_images = images[:, :, 10]
    db_labels = labels[:10]
    db_theta = cnn_init_params.cnn_init_params(
        image_dim, db_filterDim, db_numFilters, db_poolDim, num_classes)

    cost, grad, preds = cnn_cost.cnn_cost(db_theta, db_images, db_labels, num_classes,
                                          db_filterDim, db_numFilters, db_poolDim)


    # Check gradients
    J = lambda x: cnn_cost.cnn_cost(x, db_images, db_labels, num_classes,
                                    db_filterDim, db_numFilters, db_poolDim)
    num_grad = gradient.compute_gradient(J, db_theta)

    # Use this to visually compare the gradients side by side
    print num_grad, grad

    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    # Should be small. In our implementation, these values are usually
    # less than 1e-9.
    print diff

    assert diff < 1e-9, 'Difference too large. Check your gradient computation again'

##======================================================================
## STEP 3: Learn Parameters
#  Implement minFuncSGD.m, then train the model.

options = {}
options['epochs'] = 3
options['minibatch'] = 256
options['alpha'] = 1e-1
options['momentum'] = .95

opt_theta = min_func_sgd.min_func_sgd(
    lambda x, y, z: cnn_cost.cnn_cost(x, y, z, num_classes, filter_dim, num_filters, pool_dim),
    theta, images, labels, options)

##======================================================================
## STEP 4: Test
#  Test the performance of the trained model using the MNIST test set. Your
#  accuracy should be above 97% after 3 epochs of training

test_images = load_MNIST.load_MNIST_images('data/mnist/t10k-images-idx3-ubyte')
test_images = test_images.reshape(image_dim, image_dim, test_images.shape[1])
test_labels = load_MNIST.load_MNIST_images('data/mnist/t10k-labels-idx1-ubyte')
test_labels[test_labels == 0] = 10  # Remap 0 to 10

cost, grad, preds = cnn_cost.cnn_cost(opt_theta, test_images, test_labels, num_classes,
                                      filter_dim, num_filters, pool_dim, True)

acc = np.mean(preds == test_labels)

# Accuracy should be around 97.4% after 3 epochs
print('Accuracy: {0:0.3f}\n'.format(acc * 100))

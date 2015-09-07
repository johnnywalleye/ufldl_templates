import cnn
import cnn_params_to_stack
import numpy as np


def cnn_cost(theta, images, labels, num_classes, filter_dim, num_filters, pool_dim, pred=False):
    # function [cost, grad, preds] = cnnCost(theta,images,labels,num_classes,...
    #                                 filter_dim,num_filters,pool_dim,pred)
    # Calculate cost and gradient for a single layer convolutional
    # neural network followed by a softmax layer with cross entropy
    # objective.
    #
    # Parameters:
    #  theta      -  unrolled parameter vector
    #  images     -  stores images in image_dim x image_dim x numImges
    #                array
    #  num_classes -  number of classes to predict
    #  filter_dim -  dimension of convolutional filter
    #  num_filters -  number of convolutional filters
    #  pool_dim    -  dimension of pooling area
    #  pred       -  boolean only forward propagate and return
    #                predictions

    #
    # Returns:
    #  cost       -  cross entropy cost
    #  grad       -  gradient with respect to theta (if pred==False)
    #  preds      -  list of predictions for each example (if pred==True)
    image_dim = images.shape[0]  # height/width of image
    num_images = images.shape[2]  # number of images

    ## Reshape parameters and setup gradient matrices

    # Wc is filter_dim x filter_dim x num_filters parameter matrix
    # bc is the corresponding bias (for convolutional filters)

    # Wd is num_classes x hidden_size parameter matrix where hidden_size
    # is the number of output units from the convolutional layer
    # bd is corresponding bias (for softmax)
    Wc, Wd, bc, bd = cnn_params_to_stack.cnn_params_to_stack(
        theta, image_dim, filter_dim, num_filters, pool_dim, num_classes)

    # Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
    Wc_grad = np.zeros(Wc.shape)
    Wd_grad = np.zeros(Wd.shape)
    bc_grad = np.zeros(bc.shape)
    bd_grad = np.zeros(bd.shape)

    ##======================================================================
    ## STEP 1a: Forward Propagation
    #  In this step you will forward propagate the input through the
    #  convolutional and subsampling (mean pooling) layers.  You will then use
    #  the responses from the convolution and pooling layer as the input to a
    #  standard softmax layer.

    ## Convolutional Layer
    #  For each image and each filter, convolve the image with the filter, add
    #  the bias and apply the sigmoid nonlinearity.  Then subsample the
    #  convolved activations with mean pooling.  Store the results of the
    #  convolution in activations and the results of the pooling in
    #  activations_pooled.  You will need to save the convolved activations for
    #  backpropagation.
    conv_dim = image_dim - filter_dim + 1  # dimension of convolved output
    output_dim = conv_dim / pool_dim  # dimension of subsampled output

    # conv_dim x conv_dim x num_filters x num_images tensor for storing activations
    # activations = np.zeros([conv_dim, conv_dim, num_filters, num_images])
    activations = cnn.cnn_convolve(filter_dim, num_filters, images, Wc, bc)

    # output_dim x output_dim x num_filters x num_images tensor for storing
    # subsampled activations
    # activations_pooled = np.zeros([output_dim, output_dim, num_filters, num_images])
    activations_pooled = cnn.cnn_pool(pool_dim, activations)

    ### YOUR CODE HERE ###

    # Reshape activations into 2-d matrix, hidden_size x num_images,
    # for Softmax layer

    ## Softmax Layer
    #  Forward propagate the pooled activations calculated above into a
    #  standard softmax layer. For your convenience we have reshaped
    #  activationPooled into a hidden_size x num_images matrix.  Store the
    #  results in probs.

    # num_classes x num_images for storing probability that each image belongs to
    # each class.
    probs = np.zeros([num_classes, num_images])

    ### YOUR CODE HERE ###

    ##======================================================================
    ## STEP 1b: Calculate Cost
    #  In this step you will use the labels given as input and the probs
    #  calculate above to evaluate the cross entropy objective.  Store your
    #  results in cost.

    cost = 0  # save objective into cost

    ### YOUR CODE HERE ###

    # Makes predictions given probs and returns without backpropagating errors.
    if pred:
        preds = probs.argmax(axis=0) + 1
        grad = 0
        return None, grad, preds

    ##======================================================================
    ## STEP 1c: Backpropagation
    #  Backpropagate errors through the softmax and convolutional/subsampling
    #  layers.  Store the errors for the next step to calculate the gradient.
    #  Backpropagating the error w.r.t the softmax layer is as usual.  To
    #  backpropagate through the pooling layer, you will need to upsample the
    #  error with respect to the pooling layer for each filter and each image.
    #  Use the kron function and a matrix of ones to do this upsampling
    #  quickly.

    ### YOUR CODE HERE ###

    ##======================================================================
    ## STEP 1d: Gradient Calculation
    #  After backpropagating the errors above, we can use them to calculate the
    #  gradient with respect to all the parameters.  The gradient w.r.t the
    #  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
    #  a filter in the convolutional layer, convolve the backpropagated error
    #  for that filter with each image and aggregate over images.

    ### YOUR CODE HERE ###

    ## Unroll gradient into grad vector for minFunc
    grad = np.array([Wc_grad.ravel(), Wd_grad.ravel, bc_grad.ravel(), bd_grad.ravel])
    if not pred:
        preds = None
    return cost, grad, preds

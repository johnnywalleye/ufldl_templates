import numpy as np


def cnn_init_params(image_dim, filter_dim, num_filters, pool_dim, num_classes):
    # Initialize parameters for a single layer convolutional neural
    # network followed by a softmax layer.
    #
    # Parameters:
    #  image_dim  -  height/width of image
    #  filter_dim -  dimension of convolutional filter
    #  num_filters -  number of convolutional filters
    #  pool_dim    -  dimension of pooling area
    #  num_classes -  number of classes to predict
    #
    #
    # Returns:
    #  theta      -  unrolled parameter vector with initialized weights

    # Initialize parameters randomly based on layer sizes.
    assert filter_dim < image_dim, 'filter_dim must be less that image_dim'

    Wc = 1e-1 * np.random.normal(size=[filter_dim, filter_dim, num_filters])

    outDim = image_dim - filter_dim + 1  # dimension of convolved image

    # assume outDim is multiple of pool_dim
    assert outDim % pool_dim == 0, 'pool_dim must divide image_dim - filter_dim + 1'

    outDim = outDim / pool_dim
    hiddenSize = (outDim ** 2) * num_filters

    # we'll choose weights uniformly from the interval [-r, r]
    r = np.sqrt(6) / np.sqrt(num_classes + hiddenSize + 1)
    Wd = np.random.random([num_classes, hiddenSize]) * 2 * r - r

    bc = np.zeros([num_filters, 1])
    bd = np.zeros([num_classes, 1])

    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all
    # your parameters into a vector, which can then be used with minFunc.
    theta = np.concatenate([Wc.ravel(), Wd.ravel(), bc.ravel(), bd.ravel()])
    return theta

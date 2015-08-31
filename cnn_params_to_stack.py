def cnn_params_to_stack(theta, image_dim, filter_dim, num_filters, pool_dim, num_classes):
    # Converts unrolled parameters for a single layer convolutional neural
    # network followed by a softmax layer into structured weight
    # tensors/matrices and corresponding biases
    #
    # Parameters:
    #  theta      -  unrolled parameter vectors
    #  image_dim   -  height/width of image
    #  filter_dim  -  dimension of convolutional filter
    #  num_filters -  number of convolutional filters
    #  pool_dim    -  dimension of pooling area
    #  num_classes -  number of classes to predict
    #
    #
    # Returns:
    #  Wc      -  filter_dim x filter_dim x num_filters parameter matrix
    #  Wd      -  num_classes x hidden_size parameter matrix, hidden_size is
    #             calculated as num_filters*((image_dim-filter_dim+1)/pool_dim)**2
    #  bc      -  bias for convolution layer of size num_filters x 1
    #  bd      -  bias for dense layer of size hidden_size x 1
    out_dim = (image_dim - filter_dim + 1) / pool_dim
    hidden_size = out_dim ** 2 * num_filters

    # Reshape theta
    indS = 0
    indE = filter_dim ** 2 * num_filters
    Wc = theta[indS:indE].reshape([filter_dim, filter_dim, num_filters])
    indS = indE
    indE += hidden_size * num_classes
    Wd = theta[indS:indE].reshape([num_classes, hidden_size])
    indS = indE
    indE = indE + num_filters
    bc = theta[indS:indE]
    bd = theta[indE:]

    return Wc, Wd, bc, bd

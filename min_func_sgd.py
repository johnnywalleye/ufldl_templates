import numpy as np


def min_func_sgd(fun_obj, theta, data, labels, options):
    # Runs stochastic gradient descent with momentum to optimize the
    # parameters for the given objective.
    #
    # Parameters:
    #  fun_obj     -  function handle which accepts as input theta,
    #                data, labels and returns cost and gradient w.r.t
    #                to theta.
    #  theta      -  unrolled parameter vector
    #  data       -  stores data in m x n x numExamples tensor
    #  labels     -  corresponding labels in numExamples x 1 vector
    #  options    -  struct to store specific options for optimization
    #
    # Returns:
    #  opt_theta   -  optimized parameter vector
    #
    # Options (* required)
    #  epochs*     - number of epochs through data
    #  alpha*      - initial learning rate
    #  minibatch*  - size of minibatch
    #  momentum    - momentum constant, defualts to 0.9


    ##======================================================================
    ## Setup
    opt_keys = options.keys()
    assert 'epochs' not in opt_keys or 'alpha' not in opt_keys or 'minibatch' not in opt_keys, \
        'Some options not defined'
    if 'momentum' not in opt_keys:
        opt_keys['momentum'] = 0.9

    epochs = options['epochs']
    alpha = options['alpha']
    minibatch = options['minibatch']
    m = len(labels)  # training set size
    # Setup for momentum
    mom = 0.5
    momIncrease = 20
    velocity = np.zeros(len(theta))

    ##======================================================================
    ## SGD loop
    it = 0
    for e in range(epochs):

        # randomly permute indices of data for quick minibatch sampling
        rp = np.random.permutation(m)

        for s in np.arange(0, m, minibatch):  # TODO: check these ranges
            it += 1

            # increase momentum after momIncrease iterations
            if it == momIncrease:
                mom = options['momentum']

            # get next randomly selected minibatch
            mb_data = data[:, :, rp[s:s + minibatch - 1]]
            mb_labels = labels[rp[s:s + minibatch - 1]]

            # evaluate the objective function on the next minibatch
            cost, grad, _ = fun_obj(theta, mb_data, mb_labels)

            # Instructions: Add in the weighted velocity vector to the
            # gradient evaluated above scaled by the learning rate.
            # Then update the current weights theta according to the
            # sgd update rule

            ### YOUR CODE HERE ###

            print 'Epoch %d: Cost on iteration %d is %f\n', e, it, cost

        # anneal learning rate by factor of two after each epoch
        alpha /= 2.0

    opt_theta = theta
    return opt_theta


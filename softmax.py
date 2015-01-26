import numpy as np
import scipy.optimize
import scipy.sparse


def softmax_cost(theta, num_classes, input_size, lambda_, data, labels):
    # num_classes: the number of classes
    # inputSize - the size N of the input vector
    # lambda_ - weight decay parameter
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    # labels - an M x 1 matrix containing the labels corresponding for the input data

    # Unroll the parameters from theta

    theta = theta.reshape(num_classes, input_size)
    num_cases = data.shape[1]
    ground_truth = np.array(scipy.sparse.csr_matrix(
        (np.ones(num_cases), (range(num_cases), labels - 1))).todense())

    # ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Compute the cost and gradient for softmax regression.
    #                You need to compute thetagrad and cost.
    #                The groundTruth matrix might come in handy.
    cost = 0.
    theta_grad = np.zeros([num_classes, input_size])






    # ------------------------------------------------------------------
    # Unroll the gradient matrices into a vector for minFunc
    return cost, theta_grad.ravel()


def softmax_train(input_size, num_classes, lambda_, data, labels, options={'maxiter': 400, 'disp': True}):
    #softmaxTrain Train a softmax model with the given parameters on the given
    # data. Returns softmaxOptTheta, a vector containing the trained parameters
    # for the model.
    #
    # input_size: the size of an input vector x^(i)
    # num_classes: the number of classes
    # lambda_: weight decay parameter
    # input_data: an N by M matrix containing the input data, such that
    #            inputData(:, c) is the cth input
    # labels: M by 1 matrix containing the class labels for the
    #            corresponding inputs. labels(c) is the class label for
    #            the cth input
    # options (optional): options
    #   options.maxIter: number of iterations to train for

    # Initialize theta randomly
    theta = 0.005 * np.random.randn(num_classes * input_size)

    J = lambda x: softmax_cost(x, num_classes, input_size, lambda_, data, labels)

    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)

    print result
    # Return optimum theta, input size & num classes
    opt_theta = result.x

    return opt_theta, input_size, num_classes


def softmax_predict(softmax_model, data):
    # softmax_model - model trained using softmax_train (opt_theta, input_size, num_classes)
    # data - the N x M input matrix, where each column data(:, i) corresponds to
    #        a single test set
    #
    # Your code should produce the prediction matrix 
    # pred, where pred(i) is argmax_c P(y(c) | x(i)).
     
    # Unroll the parameters from theta
    theta = softmax_model[0]  # this provides a num_classes x input_size matrix
    pred = np.zeros([1, data.shape[1]])  # JS: check these dimensions, not yet 100% sure they're correct!
    
    # ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Compute pred using theta assuming that the labels start 
    #                from 1.






    return pred

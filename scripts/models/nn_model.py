import numpy as np
from utils.lr_utils import initialize_parameters
from calculations.forward_propagation import L_model_forward
from calculations.compute_cost import compute_cost
from calculations.backward_propagation import L_model_backward
from calculations.update_parameters import update_parameters


def nn_model(X, Y, n_h, num_iterations=10000, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- number of gradient descent iterations
    learning_rate -- learning rate for gradient descent
    print_cost -- if True, print cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model
    """
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters, hidden_activation="tanh")
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches, hidden_activation="tanh")
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

import numpy as np
from activations.sigmoid import sigmoid
from activations.relu import relu
from activations.tanh import tanh


def L_model_forward(X, parameters, hidden_activation="relu"):
    """
    Implement forward propagation for [LINEAR->hidden_activation]*(L-1)->LINEAR->SIGMOID

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters or initialize_parameters_deep
    hidden_activation -- activation for hidden layers: "relu" or "tanh" (default: "relu")

    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches from each linear_activation_forward call
    """
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev,
            parameters['W' + str(l)],
            parameters['b' + str(l)],
            activation=hidden_activation
        )
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A,
        parameters['W' + str(L)],
        parameters['b' + str(L)],
        activation="sigmoid"
    )
    caches.append(cache)

    return AL, caches


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement forward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    A_prev -- activations from previous layer (size of previous layer, number of examples)
    W -- weights matrix (size of current layer, size of previous layer)
    b -- bias vector (size of current layer, 1)
    activation -- "sigmoid", "relu", or "tanh"

    Returns:
    A -- post-activation value
    cache -- tuple (linear_cache, activation_cache)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "tanh":
        A = tanh(Z)
    else:
        raise ValueError(f"Unsupported activation '{activation}'. Use 'sigmoid', 'relu', or 'tanh'.")

    activation_cache = Z
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation: Z = W·A + b

    Arguments:
    A -- activations from previous layer (size of previous layer, number of examples)
    W -- weights matrix (size of current layer, size of previous layer)
    b -- bias vector (size of current layer, 1)

    Returns:
    Z -- pre-activation parameter
    cache -- tuple (A, W, b)
    """
    Z = W.dot(A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

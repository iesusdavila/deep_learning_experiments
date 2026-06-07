import numpy as np
from activations.backwards import sigmoid_backward, relu_backward, tanh_backward

def L_model_backward(AL, Y, caches, hidden_activation="relu"):
    """
    Implement backward propagation for [LINEAR->hidden_activation]*(L-1)->LINEAR->SIGMOID

    Arguments:
    AL -- probability vector, output of L_model_forward
    Y -- true label vector
    caches -- list of caches from L_model_forward
    hidden_activation -- activation used for hidden layers: "relu" or "tanh" (default: "relu")

    Returns:
    grads -- dictionary with gradients dA, dW, db for each layer
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)],
            current_cache,
            activation=hidden_activation
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def linear_activation_backward(dA, cache, activation):
    """
    Implement backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer
    cache -- tuple (linear_cache, activation_cache)
    activation -- "sigmoid", "relu", or "tanh"

    Returns:
    dA_prev, dW, db
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    else:
        raise ValueError(f"Unsupported activation '{activation}'. Use 'sigmoid', 'relu', or 'tanh'.")

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer.

    Arguments:
    dZ -- gradient of the cost w.r.t. the linear output of current layer
    cache -- tuple (A_prev, W, b) from the forward pass

    Returns:
    dA_prev, dW, db
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

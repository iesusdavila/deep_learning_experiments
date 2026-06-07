import numpy as np

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]
    
    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))    
    cost = np.squeeze(cost)

    return cost
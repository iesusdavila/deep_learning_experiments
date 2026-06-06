import numpy as np

# def compute_cost(A2, Y):
#     """
#     Computes the cross-entropy cost given in equation
    
#     Arguments:
#     A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
#     Y -- "true" labels vector of shape (1, number of examples)

#     Returns:
#     cost -- cross-entropy cost given equation
    
#     """
#     m = Y.shape[1]

#     # Compute the cross-entropy cost
#     logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
#     cost = -np.sum(logprobs) / m

#     cost = float(np.squeeze(cost)) 

#     return cost

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
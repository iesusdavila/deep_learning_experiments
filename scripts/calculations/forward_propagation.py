import numpy as np
from activations.sigmoid import sigmoid
from activations.softmax import softmax
from activations.elu import elu
from activations.relu import relu
from activations.selu import selu
from activations.swish import swish
from activations.tanh import tanh

def forward_propagation(X, parameters, activation=[tanh, sigmoid]):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = activation[0](Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = activation[1](Z2)
        
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1, "A1": A1,
             "Z2": Z2, "A2": A2}
    
    return A2, cache
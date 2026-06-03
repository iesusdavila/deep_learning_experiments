import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.atleast_1d(x)
    result = np.maximum(0, x).astype(float)
    return result
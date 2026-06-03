import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    elif x.ndim == 2:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    else:
        raise ValueError("Input must be a 1D or 2D array.")
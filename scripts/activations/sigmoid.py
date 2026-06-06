import numpy as np

def sigmoid(x):
    """
    Compute the sigmoid of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute element-wise sigmoid.
    """
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

# t_x = np.array([1, 2, 3])
# print(sigmoid(t_x))  # Expected output: [0.73105858 0.88079708 0.95257413]
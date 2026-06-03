import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.array(x)
    return np.tanh(x).astype(float)

# [ 0.46211716 -0.46211716  0.90514825 -0.90514825  0.9866143  -0.9866143 ]
print(tanh([0.5, -0.5, 1.5, -1.5, 3.0, -3.0]))
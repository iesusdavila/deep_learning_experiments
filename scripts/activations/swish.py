import numpy as np

def swish(x):
    """
    Swish activation function with numerical stability.
    """
    x = np.asarray(x, dtype=float)

    x_clipped = np.clip(x, -40.0, 40.0)

    sigmoid = 1.0 / (1.0 + np.exp(-x_clipped))
    swish = x * sigmoid
    print(swish)
    return swish

# [ 6.99362264e+00 -6.37735836e-03  7.99731720e+00 -2.68280104e-03 8.99888945e+00 -1.11055118e-03]
print(swish([6.99362264e+00, -6.37735836e-03, 7.99731720e+00, -2.68280104e-03, 8.99888945e+00, -1.11055118e-03]))
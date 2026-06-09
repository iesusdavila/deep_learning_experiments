import numpy as np
from computer_vision.zero_padding import zero_pad

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer,
              numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from hparameters
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    # Compute the dimensions of the CONV output volume
    n_H = ((n_H_prev - f + 2 * pad) // stride) + 1
    n_W = ((n_W_prev - f + 2 * pad) // stride) + 1

    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):                        # loop over training examples
        a_prev_pad = A_prev_pad[i]

        for h in range(n_H):                  # loop over vertical axis
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):              # loop over horizontal axis
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):          # loop over channels/filters
                    # Slice of input volume
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Select filter weights and bias
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]

                    # Convolution step
                    s = a_slice_prev * weights
                    biases = float(biases)
                    W_sum = np.sum(s)
                    Z[i, h, w, c] = W_sum + biases

    # Save information in cache for backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache

np.random.seed(1)

a_slice_prev = np.random.randn(4, 4, 3, 3)
W = np.random.randn(4, 4, 3, 2)
b = np.random.randn(1, 1, 1, 2)
hparameters = {"pad" : 2, "stride": 2}
Z, cache_conv = conv_forward(a_slice_prev, W, b, hparameters)

print("Z's mean =\n", np.mean(Z))
print("Z[3, 2, 1] =\n", Z[3, 2, 1])
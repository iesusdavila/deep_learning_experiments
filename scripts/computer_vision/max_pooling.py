import numpy as np 

def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    X = np.array(X)
    input_height, input_width = X.shape

    if not (1 <= input_height <= 100 and 1 <= input_width <= 100):
        raise ValueError("Input dimensions must be between 1 and 100.")
    if not (1 <= pool_size <= min(input_height, input_width)):
        raise ValueError("Pool size must be between 1 and the minimum of input dimensions.")

    out_height = input_height // pool_size
    out_width = input_width // pool_size

    output = np.zeros((out_height, out_width))

    for i in range(out_height):
        for j in range(out_width):
            row_start = i * pool_size
            row_end = row_start + pool_size
            col_start = j * pool_size
            col_end = col_start + pool_size

            region = X[row_start:row_end, col_start:col_end]
            output[i, j] = np.max(region)

    return output.tolist()

# X = [[1,2,3,4],
#       [5,6,7,8],
#       [9,10,11,12],
#       [13,14,15,16]], pool_size = 2
print(max_pooling_2d(
    X=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]),
    pool_size=2
))  # Expected output: [[6.0, 8.0], [14.0, 16.0]]

# X = [[1,2,3,4,5,6],
#       [7,8,9,10,11,12],
#       [13,14,15,16,17,18],
#       [19,20,21,22,23,24],
#       [25,26,27,28,29,30],
#       [31,32,33,34,35,36]], pool_size = 3
print(max_pooling_2d(
    X=np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30],[31,32,33,34,35,36]]),
    pool_size=3
))  # Expected output: [[15.0, 18.0], [33.0, 36.0]]
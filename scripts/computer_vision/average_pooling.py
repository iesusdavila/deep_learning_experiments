import numpy as np 

def average_pooling_2d(X, pool_size):
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
            output[i, j] = np.mean(region)

    return output.tolist()

# X = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], pool_size = 2
print(average_pooling_2d(
    X=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]),
    pool_size=2
))  # Expected output: [[3.5, 5.5], [11.5, 13.5]]

# X = [[10,20],[30,40]], pool_size = 2
print(average_pooling_2d(
    X=np.array([[10,20],[30,40]]),
    pool_size=2
))  # Expected output: [[25.0]]
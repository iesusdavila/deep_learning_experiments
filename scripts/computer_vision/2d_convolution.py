import numpy as np

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    image = np.array(image)
    kernel = np.array(kernel)
    
    # Get dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    if not (1 <= image_height <= 100 and 1 <= image_width <= 100):
        raise ValueError("Image dimensions must be between 1 and 100.")
    if not (1 <= kernel_height <= image_height and 1 <= kernel_width <= image_width):
        raise ValueError("Kernel dimensions must be less than or equal to image dimensions.")
    if not (1 <= stride <= 5):
        raise ValueError("Stride must be between 1 and 5.")
    if not (0 <= padding <= 5):
        raise ValueError("Padding must be between 0 and 5.")

    # Calculate output dimensions
    out_height = ((image_height + 2 * padding - kernel_height) // stride) + 1
    out_width = ((image_width + 2 * padding - kernel_width) // stride) + 1

    # Initialize output array
    output = np.zeros((out_height, out_width))

    # Pad the image
    if padding > 0:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    else:
        padded_image = image

    # Perform convolution
    for i in range(out_height):
        for j in range(out_width):
            # Define the region of interest
            row_start = i * stride
            row_end = row_start + kernel_height
            col_start = j * stride
            col_end = col_start + kernel_width

            # Extract the region and perform element-wise multiplication and sum
            region = padded_image[row_start:row_end, col_start:col_end]
            output[i, j] = np.sum(region * kernel)

    return output.tolist()

# image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] kernel = [[1, 0], [0, 1]] stride = 1, padding = 0
print(conv2d(
    image=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    kernel=np.array([[1, 0], [0, 1]]),
    stride=1,
    padding=0
))  # Expected output: [[ 6.  8.] [12. 14.]]

# image = [[1, 2], [3, 4]] kernel = [[1, 1], [1, 1]] stride = 1, padding = 1
print(conv2d(
    image=np.array([[1, 2], [3, 4]]),
    kernel=np.array([[1, 1], [1, 1]]),
    stride=1,
    padding=1
))  # Expected output: [[ 1.  3.  2.] [ 4. 10.  6.] [ 3.  7.  4.]]

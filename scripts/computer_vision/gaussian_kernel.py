import numpy as np
def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    if not (1 <= size <= 21 and size % 2 == 1):
        raise ValueError("Size must be an odd integer between 1 and 21.")
    if not (0.1 <= sigma <= 10.0):
        raise ValueError("Sigma must be between 0.1 and 10.0.")

    # Create a grid of (x,y) coordinates
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    # Calculate the Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize the kernel so that the sum is 1
    kernel /= np.sum(kernel)

    return kernel.tolist()
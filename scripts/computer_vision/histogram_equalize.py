import numpy as np

def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    if not image or not image[0]:
        raise ValueError("Image must be a non-empty 2D list.")
    
    height = len(image)
    width = len(image[0])
    total_pixels = height * width
    
    # Compute histogram
    histogram = [0] * 256
    for i in range(height):
        for j in range(width):
            pixel_value = image[i][j]
            if not (0 <= pixel_value <= 255):
                raise ValueError("Pixel values must be in the range [0, 255].")
            histogram[pixel_value] += 1
    
    # Compute cumulative distribution function (CDF)
    cdf = [0] * 256
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]
    
    # Find minimum non-zero CDF value
    cdf_min = next(value for value in cdf if value > 0)
    
    # Special case: all pixels have the same value (no contrast)
    if cdf_min == total_pixels:
        # Map all pixels to 0 (or keep them unchanged)
        return [[0] * width for _ in range(height)]
    
    # Normalize CDF
    cdf_normalized = [(cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255 for i in range(256)]
    cdf_normalized = [int(round(value)) for value in cdf_normalized]
    
    # Map original pixel values to equalized values
    equalized_image = [[0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            equalized_image[i][j] = cdf_normalized[image[i][j]]
    
    return equalized_image

# image = [[0, 1], [2, 3]]
print(histogram_equalize([[0, 1], [2, 3]])) # Expected output: [[0, 85], [170, 255]]

# image = [[100, 100], [100, 100]]
print(histogram_equalize([[100, 100], [100, 100]])) # Expected output: [[0, 0], [0, 0]]
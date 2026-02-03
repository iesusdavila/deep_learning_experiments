import numpy as np

def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    if not image or not image[0]:
        raise ValueError("Image must be a non-empty 2D list.")
    
    height = len(image)
    width = len(image[0])
    
    histogram = [0] * 256
    
    for i in range(height):
        for j in range(width):
            pixel_value = image[i][j]
            if not (0 <= pixel_value <= 255):
                raise ValueError("Pixel values must be in the range [0, 255].")
            histogram[pixel_value] += 1
            
    return histogram

# image = [[0, 1], [1, 2]]
print(image_histogram([[0, 1], [1, 2]])) # Expected output: [1, 2, 1, 0, 0, ..., 0]

# image = [[128, 128], [128, 128]]
print(image_histogram([[128, 128], [128, 128]])) # Expected output: [0, 0, ..., 0, 4, 0, ..., 0]
import numpy as np

def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    # Convert to numpy array
    image = np.array(image)
    
    if not (1 <= image.shape[0] <= 100 and 1 <= image.shape[1] <= 100 and image.shape[2] == 3):
        raise ValueError("Image dimensions must be between 1 and 100, and have 3 color channels.")

    # Define luminance weights
    weights = np.array([0.299, 0.587, 0.114])

    # Perform weighted sum across the color channels
    grayscale_image = np.dot(image[..., :3], weights)

    return grayscale_image.tolist()

# image = [[[255, 0, 0]]]
print(color_to_grayscale(
    image=np.array([[[255, 0, 0]]])
))  # Expected output: [[76.245]]

# image = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
print(color_to_grayscale(
    image=np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
))  # Expected output: [[76.245, 149.685], [29.07, 255.0]]
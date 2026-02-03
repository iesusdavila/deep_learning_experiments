import math 
import numpy as np

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    angel_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angel_radians)
    sin_angle = math.sin(angel_radians)

    height = len(image)
    width = len(image[0])
    
    # Centro de la imagen
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    img_rotated = [[0] * width for _ in range(height)]

    for i in range(height):
        for j in range(width):
            dy = i - cy
            dx = j - cx

            src_y = int(round(cy + dy * cos_angle + dx * sin_angle))
            src_x = int(round(cx - dy * sin_angle + dx * cos_angle))

            if 0 <= src_y < height and 0 <= src_x < width:
                img_rotated[i][j] = image[src_y][src_x]

    return img_rotated

# image = [[1,2,3],[4,5,6],[7,8,9]], angle_degrees = 180
print(rotate_image([[1,2,3],[4,5,6],[7,8,9]], 180)) # Expected output: [[9,8,7],[6,5,4],[3,2,1]]

# image = [[1,2,3],[4,5,6],[7,8,9]], angle_degrees = 0
print(rotate_image([[1,2,3],[4,5,6],[7,8,9]], 0)) # Expected output: [[1,2,3],[4,5,6],[7,8,9]]
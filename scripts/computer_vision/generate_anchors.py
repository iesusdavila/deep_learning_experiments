import math 

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    assert feature_size >= 1 and feature_size >= 1, "Feature size must be positive"
    assert image_size >= 1 and image_size >= 1, "Image size must be positive"

    stride = image_size / feature_size

    anchors = []

    for y in range(feature_size):
        for x in range(feature_size):
            center_x = (x + 0.5) * stride
            center_y = (y + 0.5) * stride

            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    width = scale * math.sqrt(aspect_ratio)
                    height = scale / math.sqrt(aspect_ratio)

                    x_min = center_x - width / 2
                    y_min = center_y - height / 2
                    x_max = center_x + width / 2
                    y_max = center_y + height / 2

                    anchors.append([x_min, y_min, x_max, y_max])

    return anchors

# feature_size = 1, image_size = 8, scales = [4], aspect_ratios = [1.0]
print(generate_anchors(1, 8, [4], [1.0])) # Expected output: [(2.0, 2.0, 6.0, 6.0)]

# feature_size = 2, image_size = 8, scales = [2], aspect_ratios = [1.0]
print(generate_anchors(2, 8, [2], [1.0])) # Expected output: [(1.0, 1.0, 3.0, 3.0), (5.0, 1.0, 7.0, 3.0), (1.0, 5.0, 3.0, 7.0), (5.0, 5.0, 7.0, 7.0)]
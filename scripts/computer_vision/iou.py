def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # box = [x_min, y_min, x_max, y_max]
    x_left_b1, y_left_b1, x_right_b1, y_right_b1 = box_a

    x_left_b2, y_left_b2, x_right_b2, y_right_b2 = box_b

    x_left_inter = max(x_left_b1, x_left_b2)
    y_left_inter = max(y_left_b1, y_left_b2)
    x_right_inter = min(x_right_b1, x_right_b2)
    y_right_inter = min(y_right_b1, y_right_b2)
    print(x_left_inter)
    print(y_left_inter)
    print(x_right_inter)
    print(y_right_inter)

    if x_right_inter < x_left_inter or y_right_inter < y_left_inter:
        return 0.0
    
    intersection_area = (x_right_inter - x_left_inter) * (y_right_inter - y_left_inter)

    box_a_area = (x_right_b1 - x_left_b1) * (y_right_b1 - y_left_b1)
    box_b_area = (x_right_b2 - x_left_b2) * (y_right_b2 - y_left_b2)

    union = box_a_area + box_b_area - intersection_area

    if union == 0:
        return 0.0

    iou = intersection_area / float(union)

    return iou


# box_a = [0, 0, 4, 4] box_b = [2, 2, 6, 6]
print(iou(box_a=[0, 0, 4, 4], box_b=[2, 2, 6, 6]))  # Expected output: 0.142

# box_a = [0, 0, 2, 2] box_b = [3, 3, 5, 5]
print(iou(box_a=[0, 0, 2, 2], box_b=[3, 3, 5, 5]))  # Expected output: 0.0

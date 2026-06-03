import math

def selu(x):
    """
    Apply SELU activation to each element.
    """
    lambda_se = 1.0507009873554804934193349852946
    alpha_se = 1.6732632423543772848170429916717

    if len(x) < 1:
        raise ValueError("Input array must have at least one element.")

    result = []
    for value in x:
        if value >= 0:
            result.append(float(lambda_se * value))
        else:
            result.append(float(lambda_se * (alpha_se * (math.exp(value) - 1))))

    return result
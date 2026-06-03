import math

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    if len(x) == 0:
        raise ValueError("Input array must have at least one element.")
    
    assert alpha >= 0, "Alpha must be positive"

    result = []
    for value in x:
        if value > 0:
            result.append(float(value))
        else:
            result.append(float(alpha * (math.exp(value) - 1)))

    return result
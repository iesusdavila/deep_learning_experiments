import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    if lr <= 0.0:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if eps <= 0.0:
        raise ValueError("Invalid epsilon value: {}".format(eps))
    
    dimension = len(w)
    if dimension < 1 or dimension > 100000:
        raise ValueError("Parameter dimension D out of bounds: {}".format(dimension))

    w = np.array(w)
    g = np.array(g)
    G = np.array(G)
    
    # Accumulate squared gradients
    new_g = G + g ** 2
    
    # Update parameters
    w_new = w - lr * g / (np.sqrt(new_g) + eps)
    
    return w_new, new_g

# Input: w=[1.0, 2.0], g=[0.1, -0.2], G=[0.0, 0.0], lr=0.1
print(adagrad_step(w=[1.0, 2.0], g=[0.1, -0.2], G=[0.0, 0.0], lr=0.1))

# Input: w=[1.0, 2.0], g=[0.0, 0.0], G=[0.1, 0.2], lr=0.1
print(adagrad_step(w=[1.0, 2.0], g=[0.0, 0.0], G=[0.1, 0.2], lr=0.1))

# Input: w=[0.0], g=[1.0], G=[100.0], lr=0.1
print(adagrad_step(w=[0.0], g=[1.0], G=[100.0], lr=0.1))
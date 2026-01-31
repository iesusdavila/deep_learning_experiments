import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    if beta1 < 0.0 or beta1 >= 1.0:
        raise ValueError("Invalid beta1 parameter: {}".format(beta1))
    if beta2 < 0.0 or beta2 >= 1.0:
        raise ValueError("Invalid beta2 parameter: {}".format(beta2))
    if lr <= 0.0:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if weight_decay < 0.0:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
    
    w = np.array(w)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    w_new = w - lr * (weight_decay * w) - lr * m / (np.sqrt(v) + eps)

    return w_new, m, v

# Input:         w=[1.0, -2.0], m=[0.0, 0.0], v=[0.0, 0.0], grad=[0.3, -0.7], lr=0.01, weight_decay=0.1
print(adamw_step(w=[1.0, -2.0], m=[0.0, 0.0], v=[0.0, 0.0], grad=[0.3, -0.7], lr=0.01, weight_decay=0.1))

# Input:         w=[1.0, 2.0], m=[0.1, 0.2], v=[0.01, 0.04], grad=[0.0, 0.0], lr=0.01, weight_decay=0.1
print(adamw_step(w=[1.0, 2.0], m=[0.1, 0.2], v=[0.01, 0.04], grad=[0.0, 0.0], lr=0.01, weight_decay=0.1))

# Input:         w=[1.0, 2.0], m=[0.1, 0.2], v=[0.01, 0.04], grad=[0.1, 0.2], lr=0.01, weight_decay=0.0
print(adamw_step(w=[1.0, 2.0], m=[0.1, 0.2], v=[0.01, 0.04], grad=[0.1, 0.2], lr=0.01, weight_decay=0.0))

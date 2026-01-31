import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * grad
    # Update biased second raw moment estimate
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected first moment estimate
    m_hat = m_new / (1 - beta1 ** t)
    # Compute bias-corrected second raw moment estimate
    v_hat = v_new / (1 - beta2 ** t)
    
    # Update parameters
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return param_new, m_new, v_new

# Input: grad=0
print(adam_step(param=np.array([1.0, 2.0]), grad=np.array([0.0, 0.0]), m=np.array([0.0, 0.0]), v=np.array([0.0, 0.0]), t=1))

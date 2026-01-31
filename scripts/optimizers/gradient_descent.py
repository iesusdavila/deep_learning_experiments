def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    f(x) = a*x^2 + b*x + c
    df/dx = 2*a*x + b (gradient)
    """
    x = x0
    for _ in range(steps):
        grad = 2 * a * x + b
        x = x - lr * grad
    return x

# Input:                         a = 1, b = -4, c = 3, x0 = 0, lr = 0.1, steps = 50
print(gradient_descent_quadratic(a = 1, b = -4, c = 3, x0 = 0, lr = 0.1, steps = 50))  # Expected output: close to 2

# Input:                         a = 0.5, b = -1, c = 0, x0 = -5, lr = 0.2, steps = 100
print(gradient_descent_quadratic(a = 0.5, b = -1, c = 0, x0 = -5, lr = 0.2, steps = 100))  # Expected output: close to 1
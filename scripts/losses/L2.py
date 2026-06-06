import numpy as np

def L2_loss(y_true, y_pred):
    """
    Compute the L2 loss (mean squared error) between true and predicted values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    
    loss = np.sum((y_true - y_pred) ** 2)

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

print("L2 = " + str(L2_loss(yhat, y)))
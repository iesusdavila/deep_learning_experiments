import numpy as np

def L1_loss(y_true, y_pred):
    """
    Compute the L1 loss (mean absolute error) between true and predicted values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    
    loss = np.sum(np.abs(y_true - y_pred))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1_loss(yhat, y)))
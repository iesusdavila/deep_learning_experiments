import numpy as np
from activations.sigmoid import sigmoid
from calculations.forward_propagation import L_model_forward


def predict(w, b, X, activation=sigmoid):
    """
    Predict labels for logistic regression (project 1).

    Arguments:
    w -- weights (n_x, 1)
    b -- bias scalar
    X -- input data (n_x, m)
    activation -- activation function (default: sigmoid)

    Returns:
    Y_prediction -- predictions vector of shape (1, m)
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = activation(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction


def predict_nn(X, parameters, y=None, hidden_activation="relu"):
    """
    Predict labels using a trained L-layer or 1-hidden-layer neural network.

    Arguments:
    X -- input data (n_x, m)
    parameters -- trained parameters dict (W1, b1, ..., WL, bL)
    y -- true labels (1, m), optional; if provided, prints accuracy
    hidden_activation -- activation used in hidden layers during training (default: "relu")

    Returns:
    p -- predictions vector of shape (1, m)
    """
    probas, _ = L_model_forward(X, parameters, hidden_activation=hidden_activation)
    p = (probas > 0.5).astype(int)

    if y is not None:
        print("Accuracy: " + str(np.sum(p == y) / X.shape[1]))

    return p

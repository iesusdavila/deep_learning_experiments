import numpy as np
from activations.sigmoid import sigmoid
from activations.softmax import softmax
from activations.elu import elu
from activations.relu import relu
from activations.selu import selu
from activations.swish import swish
from activations.tanh import tanh

def propagate(w, b, X, Y, activation=sigmoid):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    
    A = activation(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    
    cost = np.squeeze(np.array(cost))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost

# w =  np.array([[1.], [2]])
# b = 1.5

# X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
# Y = np.array([[1, 1, 0]])

# grads, cost = propagate(w, b, X, Y)

# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))
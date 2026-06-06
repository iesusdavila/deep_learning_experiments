import numpy as np
from activations.sigmoid import sigmoid
from activations.softmax import softmax
from activations.elu import elu
from activations.relu import relu
from activations.selu import selu
from activations.swish import swish
from activations.tanh import tanh

def predict(w, b, X, activation=sigmoid):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = activation(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]        
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
            
    return Y_prediction
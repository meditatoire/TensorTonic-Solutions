import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N = X.shape[0]
    D = X.shape[1]
    W = np.zeros(D)
    b = 0.0

    for i in range(steps):
        p = _sigmoid(X@W+b)
        
        delta_w = X.T @ (p - y) / N
        delta_b = np.sum(p - y) / N

        W = W - lr * delta_w
        b = b - lr * delta_b

    return (W, b)
        

        
        
        
    
    
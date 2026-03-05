import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x)
    if rng:
        rd = rng.random(x.shape)
    else:
        rd = np.random.random(x.shape)
    mask = np.where(rd>1-p, 0, 1)
    return (x * mask / (1-p) , mask / (1-p))
    
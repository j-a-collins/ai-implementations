# imports
import numpy as np
from activations import sigmoid

def sigmoid_derivative(x, *, precomputed=False):
    """
    compute the derivative of the sigmoid function for a given input x.
    
    Parameters
    ----------
    x : float or numpy array
        The input value(s) for which the derivative of the sigmoid function will be computed.
    precomputed : bool, optional
        If True, assumes ``x`` already contains the sigmoid-activated value(s).
        If False, applies ``sigmoid`` to ``x`` before differentiating. Defaults to False.
        
    Returns
    -------
    float or numpy array
        The sigmoid function derivative value(s) for the given input(s).
    """
    x = np.asarray(x)
    sigma = x if precomputed else sigmoid(x)
    return sigma * (1.0 - sigma)

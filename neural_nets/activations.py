"""
various activation functions using numpy
author: jack collins
"""

# imports
import numpy as np

def sigmoid(x):
    """
    compute the sigmoid function for a given input x.
    
    Parameters
    ----------
    x : float or numpy array
        the input values for which the sigmoid function will be computed.
        
    Returns
    -------
    float or numpy array
        the sigmoid function value(s) for the given input(s).
    """
    return 1.0 / (1 + np.exp(-x))


def relu(a: float) -> float:
    """
    computes the Rectified Linear Unit (ReLU) activation function.

    ReLU outputs the input value directly if it is positive; 
    otherwise, returns 0. it's used in neural networks to introduce 
    non-linearity while maintaining efficient gradient propagation.

    Parameters
    ----------
        a (float): Input value.

    Returns
    -------
        float: The activated value, defined as max(0.0, a).
    """
    return max(0.0, a)

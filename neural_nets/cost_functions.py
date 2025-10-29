"""
module for storing various cost functions
author: jack collins
"""

# imports
import numpy as np


def mean_squared_error(y_true, y_pred):
    """Compute mean squared error between targets and predictions.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted values produced by the model.

    Returns
    -------
    float
        The mean of the squared differences between ``y_true`` and ``y_pred``.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` do not share the same shape.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape to compute MSE."
        )

    diff = y_true - y_pred
    return np.mean(np.square(diff))

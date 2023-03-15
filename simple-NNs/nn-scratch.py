"""
A basic script for building and training a neural network from scratch,
features a feedforward process, a sum-of-squares error for the loss function,
and backpropagation. I've used gradient descent to update the weights of the 
neural network based on the gradients computed by backpropagation.
Note: the sigmoid function 'squashes' the predictions to values between 0
and 1.

Author: J-A-Collins
"""

# Imports
import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for a given input x.
    
    Parameters
    ----------
    x : float or numpy array
        The input value(s) for which the sigmoid function will be computed.
        
    Returns
    -------
    float or numpy array
        The sigmoid function value(s) for the given input(s).
    """
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Compute the derivative of the sigmoid function for a given input x.
    
    Parameters
    ----------
    x : float or numpy array
        The input value(s) for which the derivative of the sigmoid function will be computed.
    precomputed : bool, optional
        If True, assumes x is already the sigmoid value of the input. If False, calculates the sigmoid of x first.
        (default is False)
        
    Returns
    -------
    float or numpy array
        The sigmoid function derivative value(s) for the given input(s).
    """
    return x * (1.0 - x)


class NeuralNetwork:
    """A simple neural network with one hidden layer."""

    def __init__(self, x, y):
        """
        Initialise the neural network with input data, output data, and random weights.
        
        Parameters
        ----------
        x : numpy array
            The input data.
        y : numpy array
            The output data.
        """
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        """Perform the feedforward step to compute the output of the neural network."""
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        """Perform the backpropagation step to update the weights of the neural network."""
        # Application of the chain rule to find the derivative of the loss function with respect to weights2 and weights1
        output_error = 2 * (self.y - self.output)
        output_delta = output_error * sigmoid_derivative(self.output)

        d_weights2 = np.dot(self.layer1.T, output_delta)

        layer1_error = np.dot(output_delta, self.weights2.T)
        layer1_delta = layer1_error * sigmoid_derivative(self.layer1)

        d_weights1 = np.dot(self.input.T, layer1_delta)

        # Update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        return output_error


if __name__ == "__main__":
    """
    Test the NN with some dummy problem, in this case, I'll use
    a truth-table for an XOR operation: F = (A XOR B) AND C

    Table: 
    ------------------------------
       A  |   B   |   C   |   F  |
    ------+-------+-------+------+ 
    False | False | True  | False
    False | True  | True  | True
    True  | False | True  | True
    True  | True  | True  | False

    """

    # X is an array of input values
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # Y is an array
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    for i in range(1500):
        nn.feedforward()
        error = nn.backprop()
        print(f"epoch: {i}\nerror vector: \n{error}\n")

    print("final preds:")
    print(nn.output)

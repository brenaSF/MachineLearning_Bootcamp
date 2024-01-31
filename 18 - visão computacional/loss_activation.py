#Activation function:
#An activation function is a non-linear function that is applied to the output of a neuron. It introduces non-linearity into the neural network, which enables it to #learn complex patterns and relationships in the data. Without activation functions, a neural network would simply be a linear regression model. Commonly used #activation functions include sigmoid, tanh, ReLU, and softmax.

import math

# (1/(1+e^(-x)))
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# ReLU
import torch

class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(torch.zeros_like(x), x)


# Loss Functions
## Mean-squared error
import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
# Cross entropy Loss
L = - (1/N) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon) # To avoid zero division
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



















import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Numpy basics

# %% Basic Sigmoid using math.exp(x)
def basic_sigmoid(x):
    s = 1.0 / (1 + 1 / math.exp(x))
    return s

basic_sigmoid(3)

x = np.array([1, 2, 3])
print(np.exp(x))

x = np.array([1, 2, 3])
print(x+3)

# %% Basic Sigmoid using numpy
def sigmoid(x):
    s = 1.0 / (1 + 1 / np.exp(x))
    return s

x = np.array([1, 2, 3])
sigmoid(x)

# %% Sigmoid gradient
def sigmoid_derivative(x):
    s  = sigmoid(x)
    ds = s * (1-s)
    return ds

x = np.array([1, 2, 3])
print("Sigmoid derivative is " + str(sigmoid_derivative(x)))

# %% Reshaping arrays
def image2vector(image):
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    return v

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print("Image to vector is " + str(image2vector(image)))

# %% Normalizing rows
def normalizeRows(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x      = x / x_norm
    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("Normalizing rows is " + str(normalizeRows(x)))

# %% Broadcasting and the softmax function
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s     = x_exp / x_sum
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("Softmax is " + str(softmax(x)))

# %% Implementing loss functions
def L1(yhat, y):
    loss = np.sum(np.abs(yhat - y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("Loss 1 is " + str(L1(yhat, y)))

def L2(yhat, y):
    loss = np.sum(np.power((yhat - y), 2))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("Loss 2 is " + str(L2(yhat, y)))


# Logistic Regression
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

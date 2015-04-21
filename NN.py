import numpy as np
import scipy
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import math
from random import randint

OUTPUT = 10

def add_bias(np_array):
    bias = np.ones((np_array.shape[0], np_array.shape[1] + 1))
    bias[:, :-1] = np_array
    return bias

def preprocess_1(data):
    print data.shape
    if data.ndim == 1:
        data = np.append(data, 1)
        return np.reshape(data, (1, -1)) / 255.0
    with_bias = np.ones((data.shape[0], data.shape[1] + 1))
    with_bias[:, :-1] = data
    return with_bias / 255.0

def mean_squared(real, predicted):
    


class NeuralNet(object):
    labels = None
    preprocess = None
    layers = []
    cost = None

    def __init__(self, labels, data, preprocess=preprocess_1, hidden_size=200, cost_func=mean_squared):
        self.preprocess = preprocess
        self.data = preprocess(data)
        self.labels = np.zeros((len(labels), OUTPUT))
        for i, label in zip(xrange(labels.size), labels):
            self.labels[i, label] = 1
        self.layers.append(Layer(self.data.shape[1], hidden_size, True))
        self.layers.append(Layer(hidden_size + 1, OUTPUT, False))
        self.layers.append(OutputLayer())
        self.cost = cost_func
        print 'LABEL', labels[20]
        print 'ARRAY_LABEL', self.labels[20]

    def train(self):
        pass

    def forward_classify(self, data):
        z_1 = np.dot(self.preprocess(data), self.layers[0].weights)
        print self.layers[0].weights.shape
        x_1 = add_bias(self.layers[0].f(z_1))
        z_2 = np.dot(x_1, self.layers[1].weights)
        x_2 = self.layers[1].f(z_2)
        return x_2


class Layer(object):
    weights = None
    f = None

    def __init__(self, my_size, output_size, is_input=True):
        # print my_size
        np.random.seed(0)
        # self.weights = 2 * np.random.rand(my_size, output_size) - np.ones((my_size, output_size))
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=(my_size, output_size))
        if is_input:
            self.f = np.tanh
        else:
            self.f = scipy.special.expit

class OutputLayer(Layer):
    def __init__(self):
        pass

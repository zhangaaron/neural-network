import numpy as np
import scipy
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, preprocessing
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import math
from random import randint
import pickle
import os

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
    with_bias[:, :-1] = data / 255.0
    return with_bias

def mean_squared(real, predicted):
    return 0.5 * np.sum(np.square(real - predicted))


class NeuralNet(object):
    non_binary_labels = None
    labels = None
    data = None
    preprocess = None
    layers = []
    cost = None
    x_1 = None
    z_1 = None
    z_2 = None
    x_2 = None
    dJdW1 = None
    dJdW2 = None
    delta2 = None
    delta1 = None
    iteration = 0
    nu = None

    def __init__(self, labels, data, preprocess=preprocess_1, hidden_size=200, cost_func=mean_squared, nu=0.1):
        self.nu = nu
        self.preprocess = preprocess
        self.data = preprocess(data)
        self.non_binary_labels = labels
        self.labels = np.zeros((len(labels), OUTPUT))
        print labels.size
        for i, label in zip(xrange(labels.size), labels):
            self.labels[i, label] = 1
        self.layers.append(Layer(self.data.shape[1], hidden_size, True))
        self.layers.append(Layer(hidden_size + 1, OUTPUT, False))
        self.layers.append(OutputLayer())
        self.cost = cost_func
        print 'initialization done'

    def train(self):
        self.shuffle()
        for i in xrange(50000):
            if i % 50000 == 0:
                print 'performance', self.compute_accuracy()
                self.shuffle()
            self.backwards_propogate(self.data[i % 60000], self.labels[i % 60000])
            self.layers[0].weights -= self.nu * self.dJdW1
            self.layers[1].weights -= self.nu * self.dJdW2

    def forward_classify(self, data):
        self.z_1 = np.reshape(np.dot(data, self.layers[0].weights), (1, -1))
        self.x_1 = add_bias(self.layers[0].f(self.z_1))
        self.z_2 = np.dot(self.x_1, self.layers[1].weights)
        self.x_2 = self.layers[1].f(self.z_2)
        return self.x_2

    def label(self, data):
        return np.argmax(self.forward_classify(data))

    def compute_delta_2(self, data, label):
        self.delta2 = (self.x_2 - label) * self.layers[1].dFdZ(self.z_2)

    def compute_dJdW2(self, data, label):
        self.dJdW2 = self.x_1.T.dot(self.delta2)

    def compute_delta_1(self, data, label):
        right = np.dot(self.delta2, self.layers[1].weights[:-1].T)
        self.delta1 = self.layers[0].dFdZ(self.z_1) * right

    def backwards_propogate(self, data, label):
        self.forward_classify(data)
        self.compute_delta_2(data, label)
        self.compute_dJdW2(data, label)
        self.compute_delta_1(data, label)
        self.dJdW1 = np.reshape(data.T, (-1, 1)) * self.delta1
        return self.dJdW1

    def compute_accuracy(self):
        predicted = [self.label(x) for x in self.data]
        diff = self.non_binary_labels - predicted
        total = len(self.non_binary_labels)
        return float(total - np.sum(preprocessing.binarize((np.absolute(diff))))) / total

    def shuffle(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.labels)
        np.random.set_state(rng_state)
        np.random.shuffle(self.non_binary_labels)
        np.random.set_state(rng_state)
        np.random.shuffle(self.data)


def partial_tanh(z):
    return 1 - np.tanh(z) ** 2


def partial_sigmoid(z):
    sig = scipy.special.expit(z)
    return sig * (1 - sig)


class Layer(object):
    weights = None
    f = None
    dFdZ = None

    def __init__(self, my_size, output_size, is_input=True):
        # np.random.seed(0)
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(my_size, output_size))
        if is_input:
            self.f = np.tanh
            self.dFdZ = partial_tanh
        else:
            self.f = scipy.special.expit
            self.dFdZ = partial_sigmoid


class OutputLayer(Layer):
    def __init__(self):
        pass


class NNXEntropy(NeuralNet):
    eps = 0.0000001

    def compute_delta_2(self, data, label):
        cost_partial = label / (self.x_2 + self.eps) - (1 + self.eps - label) / (1 + self.eps - self.x_2)
        self.delta2 = -1 * cost_partial * self.layers[1].dFdZ(self.z_2)

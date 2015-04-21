import numpy as np
from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import math
from random import randint

def normalize(data):
    return None

class NeuralNet(object):
    labels = None
    weights = None
    preprocess = None
    layers = []
    f = None

    def __init__(self, labels, data, preprocess=normalize, hidden_size=201):
        self.preprocess = preprocess
        self.labels = np.zeros((data.shape[0], 10 ))
        for i, label in zip(xrange(labels.size), labels):
            self.labels[i, label] = 1

        # self.layers.append(Layer(self.data.shape[0], hidden_size))
        # self.layers.append(Layer(hidden_size, self.labels))
        # self.layers.append(Layer())
        # for layer in self.layers:
        #     layer.random_weights()

    def train(self, labels, data):
        pass

    def classify(self, data):
        z_1 = self.layers[0].weights * data.preprocess
        x_1 = self.f(z_1)
        z_2 = self.layers[1].weights * x_1
        x_2 = self.f(z_2)
        return x_2


class Layer(object):
    weights = None

    def __init__(self, my_size=None, output_size=None):
        self.weights = 2 * np.random.rand(my_size, output_size) - np.ones(my_size, output_size)


    def compute_values(self):
        pass

class OutputLayer(Layer):
    def __init__(self):
        pass

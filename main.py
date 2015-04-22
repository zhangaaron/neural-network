import numpy as np
from scipy import io
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import NN

def resize_data(data):
    return np.transpose(data.reshape(28 * 28, 60000))

def shuffle_and_resize(data):
    labels = data["train_labels"].ravel()
    features = resize_data(data["train_images"])
    assert len(labels) == len(features)
    #consistent shuffling src: http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    # rng_state = np.random.get_state()
    # np.random.shuffle(labels)
    # np.random.set_state(rng_state)
    # np.random.shuffle(features)
    return labels, features

data = io.loadmat("./digit-dataset/train.mat")
labels, features = shuffle_and_resize(data)


N = NN.NeuralNet(labels, features)
# print N.compute_delta_2(N.data[0], N.labels[0])
# a =  N.compute_dJdW1(N.data[0], N.labels[0])
# print a
N.train()

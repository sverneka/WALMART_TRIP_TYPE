import cPickle as pickle
import numpy as np
import scipy.sparse
import pandas as pd
import array
from sknn.mlp import Classifier, Layer, MultiLayerPerceptron


import sys
import logging

logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)


train = []
train_val = []
with open('train_short_sparse_mat.dat', 'rb') as infile:
    train = pickle.load(infile)


with open('train_val_sparse_mat.dat', 'rb') as infile:
    train_val = pickle.load(infile)


with open('label_short_sparse_mat.dat', 'rb') as infile:
    labels = pickle.load(infile)


with open('label_val_sparse_mat.dat', 'rb') as infile:
    labels_val = pickle.load(infile)


labels = labels.transpose()
labels_val = labels_val.transpose()

labels = labels.toarray()
labels_val = labels_val.toarray()

################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Tanh", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)

nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Tanh", units=200),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)

nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Tanh", units=300),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)

nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Tanh", units=100, dropout=0.5),
		Layer("Tanh", units=100, dropout=0.5),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)

nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)


nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Sigmoid", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)


nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Tanh", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)


nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Tanh", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)


nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Tanh", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)


nn.fit(train, labels)


################################# classifier 1######################
nn = Classifier(
    layers=[
        Layer("Tanh", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=50,
    batch_size=100,
    n_stable=20,
    debug=True,
	valid_set = (train_val, labels_val),
    verbose=True)


nn.fit(train, labels)

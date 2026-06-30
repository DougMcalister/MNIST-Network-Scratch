import numpy as np
import pandas as pd
import os

from read_in import (
    load_idx_images,
    load_idx_labels,
    transform_vector,
    one_hot
)
from network import Mnist_ANN, train, results_summary, plot_learning_errors

ANN_DIMS = [784, 64, 28, 10]
BATCH = 500
EPOCH = 300

TRAIN_IMG = os.path.join("data", "train-images.idx3-ubyte")
TRAIN_LBL = os.path.join("data", "train-labels.idx1-ubyte")
TEST_IMG = os.path.join("data", "t10k-images.idx3-ubyte")
TEST_LBL = os.path.join("data", "t10k-labels.idx1-ubyte")

i_train = transform_vector(load_idx_images(TRAIN_IMG))
l_train = one_hot(load_idx_labels(TRAIN_LBL))
i_test = transform_vector(load_idx_images(TEST_IMG))
l_test = one_hot(load_idx_labels(TEST_LBL))

network = Mnist_ANN(ANN_DIMS)
network.init_params()

training = train(network, i_train, l_train, EPOCH, BATCH, i_test, l_test)
results_summary(training, ANN_DIMS, EPOCH, BATCH)
plot_learning_errors(training, ANN_DIMS, EPOCH, BATCH)

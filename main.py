import os
from torch.utils.data import DataLoader
from data.dataset import MNISTdataset

from read_in import (
    load_idx_images,
    load_idx_labels,
    transform_vector,
    one_hot
)

from scratchNetwork import (
    Mnist_ANN,
    sgd
)
from data.results import (
    results_summary,
    plot_learning_errors
)

from torchNetwork import (
    cpuDevice,
    cudaDevice,
    cpuModel,
    cudaModel,
    cpuOptimiser,
    cudaOptimiser,
    run
)

ANN_DIMS = [784, 256, 128, 10]
BATCH = 128
EPOCH = 20

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

training = sgd(
    network,
    i_train,
    l_train,
    i_test,
    l_test,
    BATCH,
    EPOCH
)

results_summary(training, ANN_DIMS, EPOCH, BATCH, "Scratch CPU")
plot_learning_errors(training, ANN_DIMS, EPOCH, BATCH)

train_data = MNISTdataset(
    load_idx_images(TRAIN_IMG),
    load_idx_labels(TRAIN_LBL)
)
test_data = MNISTdataset(
    load_idx_images(TEST_IMG),
    load_idx_labels(TEST_LBL)
)

trainDataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH,
    shuffle=True
)
testDataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH
)

cpuResults = run(
    cpuModel,
    cpuDevice,
    cpuOptimiser,
    trainDataloader,
    testDataloader,
    EPOCH
)

cudaResults = run(
    cudaModel,
    cudaDevice,
    cudaOptimiser,
    trainDataloader,
    testDataloader,
    EPOCH
)
#results_summary(cpuResults, ANN_DIMS, EPOCH, BATCH, "PyTorch CPU")
#plot_learning_errors(cpuResults, ANN_DIMS, EPOCH, BATCH)
#
#results_summary(cudaResults, ANN_DIMS, EPOCH, BATCH, "PyTorch GPU")
#plot_learning_errors(cudaResults, ANN_DIMS, EPOCH, BATCH)
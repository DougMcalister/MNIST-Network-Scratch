import numpy as np

from activation import relu, softmax, cce, relu_der

SEED = 42
LR = 0.001

class Mnist_ANN:
    def __init__(self, dims):
        self.dims = dims
        self.weights = []
        self.bias = []

    def init_params(self, seed: int = SEED):
        np.random.seed(seed)
        self.weights = []
        self.bias = []

        for i in range(1, len(self.dims)):
            n_prev = self.dims[i - 1]
            n_curr = self.dims[i]

            scale = np.sqrt(2 / n_prev)
            
            w = np.random.randn(n_curr, n_prev) * scale
            b = np.zeros((n_curr, 1))

            self.weights.append(w)
            self.bias.append(b)

    def feed_forward(self, X: np.ndarray):
        A: np.ndarray = X
        self.cache = []

        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.bias[i]

            A_prev = A
            Z = W @ A + b

            if i < len(self.weights) - 1:
                A = relu(Z)
            else:
                A = softmax(Z)
            
            self.cache.append(
                {
                    "A_prev" : A_prev,
                    "A" : A,
                    "Z" : Z
                }
            )
        
        return A
    
    def back_propagation(self, y_true: np.ndarray):
        L = len(self.weights)
        m = y_true.shape[1]

        self.dw = [0.0] * L
        self.db = [0.0] * L

        y_pred = self.cache[-1]["A"]

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred has shape {y_pred.shape}, "
                f"but y_true has shape {y_true.shape}"
            )
        
        
        dZ = y_pred - y_true

        for i in reversed(range(L)):
            A_prev = self.cache[i]["A_prev"]

            self.dw[i] = (1 / m) * (dZ @ A_prev.T)
            self.db[i] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            if 0 < i:
                W = self.weights[i]
                Z_prev = self.cache[i - 1]["Z"]

                dA_prev = W.T @ dZ
                dZ = dA_prev * relu_der(Z_prev)

def data_split(n: np.ndarray):
    c = round(0.75 * len(n))
    return n[:c], n[c::]

def eval_loss(
        model: Mnist_ANN,
        X: np.ndarray,
        Y: np.ndarray
    ):
    pred = model.feed_forward(X.T)
    return cce(Y.T, pred)

def train(
        model: Mnist_ANN,
        train_data: np.ndarray,
        train_truth: np.ndarray,
        epoch: int,
        batch_size: int,
        test_data: np.ndarray,
        test_truth: np.ndarray
    ):
    history = {
        "train_loss" : [],
        "val_loss" : [],
        "epoch" : []
    }

    rng = np.random.default_rng(SEED)
    sample_count = train_data.shape[0]

    for epoch_index in range(epoch):
        indicies = rng.permutation(sample_count)

        for start in range(0, sample_count, batch_size):
            batch_indicies = indicies[start:start + batch_size]
            bX = train_data[batch_indicies].T
            bY = train_truth[batch_indicies].T

            model.feed_forward(bX)
            model.back_propagation(bY)
        
        history["train_loss"].append(eval_loss(model, train_data, train_truth))
        history["val_loss"].append(eval_loss(model, test_data, test_truth))
        history["epoch"].append(epoch_index + 1)
        
    return history

def sgd(
    model: Mnist_ANN,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    epochs: int
):
    rng = np.random.default_rng(SEED)
    
    m = len(X)
    theta = np.random.randn(2, 1)

    X_bias = np.c_[np.ones((m, 1), X)]
    cost_history = {
        "Training Cost" : [],
        "Testing Cost" : [],
        "Epoch" : []
    }

    for epoch in range(epochs):
        indicies = rng.permutation(m)

        X_shuffled = X_bias[indicies]
        y_shuffled = y[indicies]

        for i in range(0, m, batch_size):
            end = i + batch_size
            X_batch = X_shuffled[:, i : end]
            y_batch = y_shuffled[:, i:end]

            y_pred = model.feed_forward(X_batch)

            model.back_propagation(y_batch)

            for i in range(len(model.weights)):
                model.weights[i] -= (
                    LR * model.dw[i]
                )

                model.bias[i] -= (
                    LR * model.db[i]
                )
        
        y_pred = model.feed_forward(X)
        y_pred_test = model.feed_forward(X_test)

        train_cost = cce(y, y_pred)
        test_cost = cce(y_test, y_pred_test)

        cost_history["Training Cost"].append(train_cost)
        cost_history["Testing Cost"].append(test_cost)
        cost_history["Epoch"].append(epoch + 1)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, Cost: {train_cost}"
            )

    return cost_history

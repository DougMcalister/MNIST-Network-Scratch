import numpy as np
import os
from typing import Dict, List
import datetime as dt
import matplotlib.pyplot as plt

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
        
        for i in range(L):
            self.weights[i] -= LR * self.dw[i]
            self.bias[i] -= LR * self.db[i]

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

def results_summary(
        history: Dict[str, List],
        dimensions: List[int],
        epochs: int,
        batch_size: int,
        learning_rate = LR
    ):
    train_loss = np.asarray(history.get("train_loss", []), dtype=float)
    val_loss = np.asarray(history.get("val_loss", []), dtype=float)
    epoch_values = np.asarray(
        history.get("epoch", range(1, len(train_loss) + 1))
    )

    if train_loss.size == 0 or val_loss.size == 0:
        raise ValueError("History must contain training and validation losses.")
    if train_loss.size != val_loss.size:
        raise ValueError("Training and validation histories must have equal lengths.")
    if epoch_values.size != train_loss.size:
        epoch_values = np.arange(1, train_loss.size + 1)

    p_count = sum(
        (dimensions[i - 1] * dimensions[i]) + dimensions[i]
        for i in range(1, len(dimensions))
    )
    best_val_index = int(np.argmin(val_loss))
    final_gap = float(val_loss[-1] - train_loss[-1])

    print("MNIST ARTIFICIAL NEURAL NETWORK TRAINING SUMMARY")
    print("================================================")
    print("Network Variables")
    print(f"Dimensions: {dimensions}")
    print(f"Parameters: {p_count:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Requested Epochs: {epochs}")
    print(f"Completed Epochs: {train_loss.size}")
    print("================================================")
    print(f"Initial Training CCE: {train_loss[0]:.4f}")
    print(f"Final Training CCE:   {train_loss[-1]:.4f}")
    print(f"Initial Validation CCE: {val_loss[0]:.4f}")
    print(f"Final Validation CCE:   {val_loss[-1]:.4f}")
    print(
        f"Best Validation CCE:    {val_loss[best_val_index]:.4f} "
        f"(epoch {int(epoch_values[best_val_index])})"
    )
    print(f"Final Generalization Gap: {final_gap:.4f}")
    print("================================================")


def plot_learning_errors(
        history: Dict[str, List],
        dimensions: List[int],
        requested_epochs: int,
        batch_size: int,
        learning_rate=LR,
        save_path: str | None = None,
        show: bool = True
    ):
    train_loss = np.asarray(history.get("train_loss", []), dtype=float)
    val_loss = np.asarray(history.get("val_loss", []), dtype=float)
    epochs = np.asarray(history.get("epoch", range(1, len(train_loss) + 1)))

    if train_loss.size == 0 or val_loss.size == 0:
        raise ValueError("History must contain training and validation losses.")
    if train_loss.size != val_loss.size:
        raise ValueError("Training and validation histories must have equal lengths.")
    if epochs.size != train_loss.size:
        epochs = np.arange(1, train_loss.size + 1)

    p_count = sum(
        (dimensions[i - 1] * dimensions[i]) + dimensions[i]
        for i in range(1, len(dimensions))
    )

    best_val_index = int(np.argmin(val_loss))
    final_gap = float(val_loss[-1] - train_loss[-1])

    if save_path is None:
        os.makedirs("figs", exist_ok=True)
        save_path = os.path.join(
            "figs",
            f"learning_errors_plot_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M')}.png"
        )

    fig, (ax, ax_table) = plt.subplots(
        ncols=2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # Main learning curve plot
    ax.plot(epochs, train_loss, label="Training CCE", linewidth=2)
    ax.plot(epochs, val_loss, label="Validation CCE", linewidth=2)

    ax.scatter(
        epochs[best_val_index],
        val_loss[best_val_index],
        color="tab:red",
        zorder=3,
        label=f"Best validation epoch {int(epochs[best_val_index])}"
    )

    ax.set(
        title="Training and Validation Learning Curves",
        xlabel="Epoch",
        ylabel="Categorical Cross-Entropy"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    # Summary table data
    summary_data = [
        ["Dimensions", str(dimensions)],
        ["Parameters", f"{p_count:,}"],
        ["Learning Rate", f"{learning_rate}"],
        ["Batch Size", f"{batch_size}"],
        ["Requested Epochs", f"{requested_epochs}"],
        ["Completed Epochs", f"{train_loss.size}"],
        ["Initial Train CCE", f"{train_loss[0]:.4f}"],
        ["Final Train CCE", f"{train_loss[-1]:.4f}"],
        ["Initial Val CCE", f"{val_loss[0]:.4f}"],
        ["Final Val CCE", f"{val_loss[-1]:.4f}"],
        [
            "Best Val CCE",
            f"{val_loss[best_val_index]:.4f} @ epoch {int(epochs[best_val_index])}"
        ],
        ["Final Gap", f"{final_gap:.4f}"]
    ]

    # Hide the right-hand axis and draw table
    ax_table.axis("off")
    ax_table.set_title("Training Summary", fontweight="bold", pad=12)

    table = ax_table.table(
        cellText=summary_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


results_sumary = results_summary

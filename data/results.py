import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np
import datetime as dt
import os

LR = 0.001

def results_summary(
        history: Dict[str, List],
        dimensions: List[int],
        epochs: int,
        batch_size: int,
        nType: str,
        learning_rate = LR
    ):
    train_loss = np.asarray(history.get("Training Cost", []), dtype=float)
    val_loss = np.asarray(history.get("Testing Cost", []), dtype=float)
    epoch_values = np.asarray(
        history.get("Epoch", range(1, len(train_loss) + 1))
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
    print(f"Network Type: {nType}")
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
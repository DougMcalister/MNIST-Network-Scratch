import torch
from torch import nn
from torch.utils.data import DataLoader

cpuDevice = "cpu"
cudaDevice = (
    torch.accelerator.current_accelerator().type  #type: ignore 
    if torch.accelerator.is_available() else "cpu"
)

class NeuralNetwork (nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

cpuModel = NeuralNetwork().to(cpuDevice)
cudaModel = NeuralNetwork().to(cudaDevice)

loss_fn = nn.CrossEntropyLoss()

cpuOptimiser = torch.optim.SGD(
    cpuModel.parameters(),
    lr = 0.001
)
cudaOptimiser = torch.optim.SGD(
    cudaModel.parameters(),
    lr = 0.001
)

def train(
    dataloader: DataLoader,
    device,
    model: NeuralNetwork,
    optimiser: torch.optim.SGD,
    loss_fn = loss_fn
):
    model.train()

    total_loss = 0.0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        optimiser.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()

        optimiser.step()

        total_loss += loss

    return total_loss / len(dataloader), total_loss

def test(
    dataloader: DataLoader,
    device,
    model: NeuralNetwork,
    loss_fn = loss_fn
):
    model.eval()

    total_loss = 0 ; correct = 0 ; total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            loss = loss_fn(pred, y)

            total_loss += loss.item()

            predicted_class = pred.argmax(dim=1)

            correct += (
                predicted_class == y
            ).sum().item()

            total += y.size(0)
    average_loss = (
        total_loss / len(dataloader)
    )

    accuracy = correct / total

    return average_loss, accuracy, total_loss

def run(
    model: NeuralNetwork,
    device,
    optimiser: torch.optim.SGD,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int
):
    cost_history = {
        "Training Cost" : [],
        "Testing Cost" : [],
        "Epoch" : []
    }

    for epoch in range(epochs):
        mean_train_loss, total_train_loss = train(
            train_loader,
            device, 
            model,
            optimiser
        )
        print(
            f"Training epoch {epoch + 1} complete\n"
            f"Average loss = {mean_train_loss}\n"
            f"Total loss = {total_train_loss}\n"
            "Begining testing run...\n"
        )

        mean_test_loss, acc, total_test_loss = test(
            test_loader,
            device, 
            model
        )
        print(
            f"Testing epoch {epoch + 1} complete\n"
            f"Average loss = {mean_test_loss}\n"
            f"Accuracy = {acc * 100}%\n"
            f"Total loss = {total_test_loss}\n"
        )
        if epoch + 1 != epochs:
            print(f"Begining epoch {epoch + 2}")
        else:
            print("Training complete")

        cost_history["Training Cost"].append(total_train_loss)
        cost_history["Testing Cost"].append(total_test_loss)
        cost_history["Epoch"].append(epoch + 1)
    return cost_history
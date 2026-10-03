from typing import Any

import torch

class MNISTdataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index) -> Any:
        return self.X[index], self.y[index]
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

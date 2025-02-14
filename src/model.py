import torch
import torch.nn as nn


class LITSModel(nn.Module):
    def __init__(
        self,
        board_size: int,
        num_xs: int,
        max_pieces_per_shape: int,
        num_conv_layers: int,
        num_linear_layers: int,
    ):
        if num_conv_layers == 0 and num_linear_layers == 0:
            raise ValueError("Model must have at least one layer")
        super().__init__()
        self.board_size = board_size
        self.num_xs = num_xs
        self.max_pieces_per_shape = max_pieces_per_shape
        conv_layers = []
        for i in range(num_conv_layers):
            if i == 0:
                conv_layers.append(nn.Conv2d(5, 32, 3, padding=0))
            else:
                conv_layers.append(nn.Conv2d(32, 32, 5, padding="same"))
            conv_layers.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_layers)
        linear_layers = []
        for i in range(num_linear_layers):
            if i == 0:
                if num_conv_layers:
                    linear_layers.append(nn.Linear(32 * (board_size - 2) ** 2, 512))
                else:
                    linear_layers.append(nn.Linear(5 * board_size**2, 512))
            else:
                linear_layers.append(nn.Linear(512, 512))
            linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(512, 32))
        linear_layers.append(nn.ReLU())
        linear_layers.append(nn.Linear(32, 1))
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

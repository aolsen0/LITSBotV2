import os
import json
from typing import Any, Type, TypeVar

import torch
import torch.nn as nn

from src.piece_utils import get_total_number_of_pieces

T = TypeVar("T", bound="BaseLITSModel")


class BaseLITSModel(nn.Module):
    """Base class for LITS models that provides save/load helpers.

    Subclasses should set:
    - name: str
    - save_params: dict[str, Any]  # parameters needed to reconstruct the model
    - single_output: bool  # whether the model outputs a single value or additional
        legality information
    - board_size, num_xs, max_pieces_per_shape: int  # game parameters
    """

    name: str
    save_params: dict[str, Any]
    single_output: bool
    board_size: int
    num_xs: int
    max_pieces_per_shape: int

    def __init__(self):
        super().__init__()

    def save(self, identifier: str, base_dir: str = "models") -> None:
        """Save model state_dict and metadata to models/[name]/.

        Args:
            identifier: Unique identifier for this model version, such as the number of
                training games completed.
            base_dir: Base directory to save models in.
        """
        if not hasattr(self, "name"):
            raise AttributeError("Model must define a 'name' attribute before saving.")
        if not hasattr(self, "save_params"):
            raise AttributeError(
                "Model must define a 'save_params' attribute before saving."
            )
        model_dir = os.path.join(base_dir, self.name)
        os.makedirs(model_dir, exist_ok=True)

        meta = {"class": self.__class__.__name__, "params": self.save_params}
        meta_path = os.path.join(model_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                existing_meta = json.load(f)
            if existing_meta != meta:
                raise ValueError(
                    f"Model metadata conflicts with existing metadata at {meta_path}"
                )
        else:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        state_path = os.path.join(model_dir, f"{identifier}.pt")
        torch.save(self.state_dict(), state_path)

    @classmethod
    def load(cls: Type[T], name: str, identifier: str, base_dir: str = "models") -> T:
        """Load a model from models/[name]/[identifier].pt using metadata in meta.json.

        Args:
            name: Name of the model to load.
            identifier: Unique identifier for the model version to load.
            base_dir: Base directory to load models from.
        Returns:
            Instantiated model with loaded weights (on CPU).
        """
        model_dir = os.path.join(base_dir, name)
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        class_name = meta.get("class")
        params = meta.get("params", {})

        if cls.__name__ != class_name:
            raise ValueError(f"Model {name} is a {class_name}, not a {cls.__name__}")

        model = cls(name=name, **params)

        state_path = os.path.join(model_dir, f"{identifier}.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Model state file not found at {state_path}")

        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
        return model


class LITSModel(BaseLITSModel):
    def __init__(
        self,
        name: str,
        board_size: int,
        num_xs: int,
        max_pieces_per_shape: int,
        num_conv_layers: int,
        num_linear_layers: int,
    ):
        if num_linear_layers == 0:
            raise ValueError("Model must have at least one linear layer")
        super().__init__()
        self.name = name
        self.single_output = True
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

        self.save_params = {
            "board_size": board_size,
            "num_xs": num_xs,
            "max_pieces_per_shape": max_pieces_per_shape,
            "num_conv_layers": num_conv_layers,
            "num_linear_layers": num_linear_layers,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MoveModel(BaseLITSModel):
    """Instead of simply estimating the value of a position, this model estimates the
    value of each possible move in a position, as well as the legality of each move.
    Including legality in the model allows us to evaluate positions without explicitly
    checking which moves are legal.
    """

    def __init__(
        self,
        name: str,
        board_size: int,
        num_xs: int,
        max_pieces_per_shape: int,
        num_conv_layers: int,
        num_linear_layers: int,
    ):
        if num_linear_layers == 0:
            raise ValueError("Model must have at least one layer")
        super().__init__()
        self.name = name
        self.single_output = False
        self.board_size = board_size
        self.num_xs = num_xs
        self.max_pieces_per_shape = max_pieces_per_shape
        legal_moves = get_total_number_of_pieces(board_size)
        conv_layers = []
        for i in range(num_conv_layers):
            if i == 0:
                conv_layers.append(nn.Conv2d(5, 32, 3, padding=0))
            else:
                conv_layers.append(nn.Conv2d(32, 32, 5, padding="same"))
            conv_layers.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_layers)
        if num_conv_layers:
            splitter = nn.Linear(32 * (board_size - 2) ** 2, 2 * legal_moves)
        else:
            splitter = nn.Linear(5 * board_size**2, 2 * legal_moves)
        self.splitter = nn.Sequential(splitter, nn.ReLU())

        linear_layers = []
        for i in range(num_linear_layers):
            linear_layers.append(nn.Linear(legal_moves, legal_moves))
            if i < num_linear_layers - 1:
                linear_layers.append(nn.ReLU())
        self.linear = nn.Sequential(*linear_layers)

        legal_layers = [
            nn.Linear(legal_moves, legal_moves),
            nn.ReLU(),
            nn.Linear(legal_moves, legal_moves),
            nn.Sigmoid(),
        ]
        self.legal = nn.Sequential(*legal_layers)

        self.save_params = {
            "board_size": board_size,
            "num_xs": num_xs,
            "max_pieces_per_shape": max_pieces_per_shape,
            "num_conv_layers": num_conv_layers,
            "num_linear_layers": num_linear_layers,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.splitter(x).view(x.size(0), 2, -1)
        value = x[:, 0, :]
        legal = x[:, 1, :]
        x = self.linear(value)
        legal = self.legal(legal)
        return torch.stack([x, legal], dim=1)

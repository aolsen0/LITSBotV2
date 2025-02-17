import torch
import torch.nn as nn
from src.game import LITSGame
from src.model import LITSModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model: LITSModel, games: int, epsilon: float, lr: float) -> None:
    """Train a model to play Battle of LITS.

    Args:
        model: The model to train.
        games: The number of games to play during training.
        epsilon: The probability of choosing a random move instead of the best move.
        lr: The learning rate for the optimizer.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    recent_losses = [0.0] * 1000
    for _ in range(games):
        game = LITSGame(
            board_size=model.board_size,
            num_xs=model.num_xs,
            max_pieces_per_shape=model.max_pieces_per_shape,
        )
        model.train()
        inputs, value = game.generate_examples(model, epsilon)
        optimizer.zero_grad()
        output = model(inputs.to(device))
        loss = loss_fn(output, value.to(device))
        recent_losses[_ % 1000] = loss.item()
        if _ % 1000 == 999:
            print(f"{_ + 1} games played. Loss: {sum(recent_losses) / 1000}")
        loss.backward()
        optimizer.step()

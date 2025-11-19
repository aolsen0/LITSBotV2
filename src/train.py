import os
import time
import torch
import torch.nn as nn
from src.game import LITSGame
from src.model import BaseLITSModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: BaseLITSModel,
    epsilon: float,
    lr: float,
    start_games: int,
    train_games: int,
    save_interval: int | None = None,
    output_interval: int = 1000,
) -> None:
    """Train a model to play Battle of LITS.

    Args:
        model: The model to train.
        epsilon: The probability of choosing a random move instead of the best move.
        lr: The learning rate for the optimizer.
        start_games: The number of games the model has already been trained on.
        train_games: The number of additional games to train the model on.
        save_interval: If provided, the number of games between saving the model.
        output_interval: The number of games between printing loss information.
    """
    model_dir = model.get_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    training_log_path = os.path.join(model_dir, "training_log.txt")
    if not os.path.exists(training_log_path):
        with open(training_log_path, "w") as log_file:
            pass

    def log(message: str) -> None:
        with open(training_log_path, "a") as log_file:
            log_file.write(message + "\n")

    log(f"Training started at {time.ctime()} with lr={lr}, epsilon={epsilon}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if model.single_output:
        loss_fn = nn.MSELoss()
    else:

        def loss_fn(output, target):
            weights = torch.full_like(target, 1.0)
            # we do not care about the value of illegal moves
            weights[:, 0] = torch.where(target[:, 1] == 0, 0.0, 1.0)
            return nn.functional.mse_loss(output, target, weight=weights)

    recent_losses = []
    start_time = time.time()
    for game_num in range(start_games + 1, start_games + train_games + 1):
        game = LITSGame(
            board_size=model.board_size,
            num_xs=model.num_xs,
            max_pieces_per_shape=model.max_pieces_per_shape,
        )
        model.eval()
        inputs, value = game.generate_examples(model, epsilon)
        model.train()
        optimizer.zero_grad()
        output = model(inputs.to(device))
        loss = loss_fn(output, value.to(device))
        recent_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if game_num % output_interval == 0:
            elapsed_time = time.time() - start_time
            line = (
                f"{elapsed_time:.2f} seconds elapsed. "
                f"{game_num} games played. Loss: {sum(recent_losses) / len(recent_losses)}"
            )
            print(line)
            log(line)
            recent_losses = []
        if save_interval is not None and game_num % save_interval == 0:
            identifier = f"{game_num}"
            model.save(identifier)
            log(f"Model saved at game {game_num}")

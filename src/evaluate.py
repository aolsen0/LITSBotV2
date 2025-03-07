import torch
from src.game import LITSGame
from src.model import LITSModel, MoveModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_models(
    model1: LITSModel | MoveModel, model2: LITSModel | MoveModel, num_games: int = 1
) -> int:
    """Compare two models by playing several games against each other.

    Args:
        model1: The first model to compare.
        model2: The second model to compare.
    """
    if (
        model1.board_size != model2.board_size
        or model1.num_xs != model2.num_xs
        or model1.max_pieces_per_shape != model2.max_pieces_per_shape
    ):
        raise ValueError("Models must have the  same game parameters")
    model1 = model1.to(device)
    model2 = model2.to(device)
    wins = 0
    for i in range(num_games):
        game = LITSGame(
            board_size=model1.board_size,
            num_xs=model1.num_xs,
            max_pieces_per_shape=model1.max_pieces_per_shape,
        )
        if i % 2 == 0:
            first, second = model1, model2
        else:
            first, second = model2, model1
        while not game.completed:
            if game.current_player == 0:
                if isinstance(first, LITSModel):
                    game.play_best(first)
                elif isinstance(first, MoveModel):
                    game.play_best(first, False)
                else:
                    raise ValueError("Invalid model type")
            else:
                if isinstance(second, LITSModel):
                    game.play_best(second)
                elif isinstance(second, MoveModel):
                    game.play_best(second, False)
                else:
                    raise ValueError("Invalid model type")
        if game.score() > 0 and i % 2 == 0 or game.score() < 0 and i % 2 == 1:
            wins += 1
    return wins

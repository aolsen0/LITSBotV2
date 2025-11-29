import torch
from src.game import LITSGame
from src.model import BaseLITSModel, MoveModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_models(
    model1: BaseLITSModel, model2: BaseLITSModel, num_games: int = 1
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
        raise ValueError("Models must have the same game parameters")
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
                game.play_best(first)
            else:
                game.play_best(second)
        if game.score() > 0 and i % 2 == 0 or game.score() < 0 and i % 2 == 1:
            wins += 1
    return wins


def compare_think(
    model1: BaseLITSModel,
    model2: BaseLITSModel,
    num_games: int = 1,
    seconds_per_move: float = 5.0,
) -> int:
    """Compare two models by playing games with alpha-beta search.

    Args:
        model1: The first model to compare.
        model2: The second model to compare.
        num_games: The number of games to play.
        seconds_per_move: The number of seconds of calculation to allow for each move.
    """
    if (
        model1.board_size != model2.board_size
        or model1.num_xs != model2.num_xs
        or model1.max_pieces_per_shape != model2.max_pieces_per_shape
    ):
        raise ValueError("Models must have the same game parameters")
    model1 = model1.to(device)
    model2 = model2.to(device)
    total_score = 0
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
                game.play_think(first, seconds_per_move, quiet=True)
            else:
                game.play_think(second, seconds_per_move, quiet=True)
        # don't actually reward for tiebreak wins (which have score of +/-0.5)
        if i % 2 == 0:
            total_score += int(game.score())
        else:
            total_score -= int(game.score())
    return total_score / num_games


def legality_accuracy(
    model: MoveModel, games: int = 1000, epsilon: float = 0.0
) -> None:
    """Calculate the accuracy of a model's legality predictions.

    Args:
        model: The model to evaluate.
        games: The number of games to play to evaluate the model.
    """
    model = model.to(device)
    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0
    total_loss = 0.0
    for _ in range(games):
        game = LITSGame(
            board_size=model.board_size,
            num_xs=model.num_xs,
            max_pieces_per_shape=model.max_pieces_per_shape,
        )
        inputs, outputs = game.generate_examples(model, epsilon)
        result = model(inputs.to(device))[:, 1]
        legal = outputs[:, 1]
        true_positives += torch.sum((result > 0.5) & (legal > 0.5)).item()
        true_negatives += torch.sum((result <= 0.5) & (legal <= 0.5)).item()
        false_negatives += torch.sum((result <= 0.5) & (legal > 0.5)).item()
        false_positives += torch.sum((result > 0.5) & (legal <= 0.5)).item()
        total_loss += torch.mean((result - legal) ** 2).item()
    total = true_positives + true_negatives + false_negatives + false_positives
    print(f"Accuracy: {(true_positives + true_negatives) / total}")
    print(f"Precision: {true_positives / (true_positives + false_positives)}")
    print(f"Recall: {true_positives / (true_positives + false_negatives)}")
    print(f"MSE Loss: {total_loss / games}")
    print(f"True positives: {true_positives}, False positives: {false_positives}")
    print(f"True negatives: {true_negatives}, False negatives: {false_negatives}")

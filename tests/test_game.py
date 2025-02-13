from unittest.mock import patch
import pytest
import torch
from src.game import LITSGame


def test_game_play():
    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)
    game.play(0)
    assert game.board.played_ids == [0]
    assert game.current_player == 1

    with pytest.raises(ValueError):
        game.play(1)

    game.play(-1)
    assert game.current_player == 0
    assert game.swapped
    assert game.board.played_ids == [0]

    game.play(72)
    assert game.board.played_ids == [0, 72]
    assert game.completed

    with pytest.raises(ValueError):
        game.play(0)


def test_game_score():
    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)
    game.board._board_tensor = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    assert game.score() == 0.0
    game.play(0)
    assert game.score() == 2.0
    game.play(-1)
    assert game.score() == -2.0
    game.play(98)
    assert game.score() == 0.5

    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)
    game.board._board_tensor = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    game.play(0)
    game.play(72)
    assert game.score() == 1.0


@patch("builtins.input")
def test_game_prompt(input_mock):
    input_mock.side_effect = [
        "A 1 B 1 C 1 D 1",
        "n",
        "A 1 B 1 C 1 D 1",
        "B 2 B 3 B 4 C 4",
    ]
    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)
    game.prompt()
    assert game.board.played_ids == [48]
    assert game.current_player == 1

    game.prompt()
    assert not game.swapped
    assert game.board.played_ids == [48, 32]
    assert game.completed

    game.prompt()

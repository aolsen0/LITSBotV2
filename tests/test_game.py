from unittest.mock import patch
import pytest
import torch
from src.game import LITSGame
from src.piece_utils import get_stacked_piece_tensor


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


def test_generate_examples():
    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model(tensor):
        return -5 * torch.tensor([list(range(tensor.shape[0]))]).to(device).T

    real_board = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    game.board._board_tensor = real_board
    game.board._score_change = torch.tensordot(
        game.board._board_tensor, get_stacked_piece_tensor(4), dims=[[0, 1], [1, 2]]
    )

    example_in, example_out = game.generate_examples(model, 0.0)
    assert example_in.shape == (3, 5, 4, 4)
    assert example_out.shape == (3, 1)
    assert example_out.tolist() == [[45], [0], [0]]
    assert example_in[0, 0].equal(-real_board)
    assert example_in[1, 0].equal(real_board)
    assert example_in[2, 0].equal(-real_board)
    assert example_in[1, 1:].sum(axis=0).tolist() == [
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ]

    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)

    real_board = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    game.board._board_tensor = real_board
    example_in, example_out = game.generate_examples(model, 1.0)
    assert example_in[0, 0].equal(-real_board)
    assert example_in[1, 0].equal(real_board)

    # Test with single_output=False
    def model(tensor):
        return 5 * torch.tensor([list(range(tensor.shape[0] * 208))]).reshape(
            [tensor.shape[0], 2, 104]
        ).to(device)

    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)

    real_board = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    game.board._board_tensor = real_board
    game.board._score_change = torch.tensordot(
        game.board._board_tensor, get_stacked_piece_tensor(4), dims=[[0, 1], [1, 2]]
    )
    example_in, example_out = game.generate_examples(model, 0.0, False)
    assert example_in.shape == (3, 5, 4, 4)
    assert example_out.shape == (3, 2, 104)
    assert example_out[0][1].sum() == 10
    assert example_out[0][0][52] == -515
    assert example_out[2].equal(torch.zeros(2, 104).to(device))


def test_play_best():
    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model(tensor):
        return -5 * torch.tensor([list(range(tensor.shape[0]))]).to(device).T

    real_board = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    game.board._board_tensor = real_board
    game.board._score_change = torch.tensordot(
        game.board._board_tensor, get_stacked_piece_tensor(4), dims=[[0, 1], [1, 2]]
    )

    game.play_best(model)
    assert game.board.played_ids == [0]
    assert game.current_player == 1

    game.play_best(model)
    assert game.board.played_ids == [0, 100]

    # Test with single_output=False
    def model(tensor):
        return 5 * torch.tensor([list(range(tensor.shape[0] * 208))]).reshape(
            [tensor.shape[0], 2, 104]
        ).to(device)

    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)
    game.board._board_tensor = real_board
    game.board._score_change = torch.tensordot(
        game.board._board_tensor, get_stacked_piece_tensor(4), dims=[[0, 1], [1, 2]]
    )
    game.play_best(model, False)
    assert game.board.played_ids == [0]


def test_game_evaluate():
    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)

    def model(tensor):
        return torch.tensor(-5.0)

    real_board = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    game.board._board_tensor = real_board
    game.board._score_change = torch.tensordot(
        game.board._board_tensor, get_stacked_piece_tensor(4), dims=[[0, 1], [1, 2]]
    ).tolist()

    game.play(0)
    assert game.evaluate(model) == 7.0
    game.play(-1)
    assert game.evaluate(model) == -7.0
    game.play(98)
    assert game.evaluate(model) == 5.5

    game = LITSGame(board_size=4, num_xs=4, max_pieces_per_shape=1)
    legality = torch.tensor([1, 0] * 52)
    score = torch.randn(104)

    def model(tensor):
        return torch.stack([score, legality], dim=0).unsqueeze(0)

    game.board._board_tensor = real_board
    game.board._score_change = torch.tensordot(
        game.board._board_tensor, get_stacked_piece_tensor(4), dims=[[0, 1], [1, 2]]
    ).tolist()
    game.play(0)
    assert game.evaluate(model, False) == 2 - score[::2].max().item()

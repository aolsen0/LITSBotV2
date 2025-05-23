from src.board import LITSBoard
from src.model import LITSModel, MoveModel


def test_model():
    model = LITSModel(
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=2,
        num_linear_layers=2,
    )
    board = LITSBoard(board_size=12)
    tensor, _ = board.to_children_tensor(list(range(32)))
    output = model(tensor)
    assert output.shape == (32, 1)

    model = LITSModel(
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=2,
        num_linear_layers=1,
    )
    output = model(tensor)
    assert output.shape == (32, 1)

    model = LITSModel(
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=0,
        num_linear_layers=2,
    )
    output = model(tensor)
    assert output.shape == (32, 1)


def test_move_model():
    model = MoveModel(
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=2,
        num_linear_layers=2,
    )
    board = LITSBoard(board_size=12)
    tensor, _ = board.to_children_tensor(list(range(32)))
    output = model(tensor)
    assert output.shape == (32, 2, 1976)

    model = MoveModel(
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=2,
        num_linear_layers=1,
    )
    output = model(tensor)
    assert output.shape == (32, 2, 1976)

    model = MoveModel(
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=0,
        num_linear_layers=2,
    )
    output = model(tensor)
    assert output.shape == (32, 2, 1976)

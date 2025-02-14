from src.board import LITSBoard
from src.model import LITSModel


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

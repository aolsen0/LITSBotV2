from src.board import LITSBoard


def test_board_construct():
    board = LITSBoard()
    tensor = board.to_tensor()
    # the board should be rotationally symmetric
    for i in range(100):
        assert tensor[i] + tensor[99 - i] == 0.0
    assert sum(abs(board.to_tensor()[:100])) == 60
    dense_board = LITSBoard(num_xs=50)
    assert sum(abs(dense_board.to_tensor()[:100])) == 100


def test_board_play():
    board = LITSBoard()
    board.play(0)
    tensor = board.to_tensor()
    assert tensor[100] == tensor[101] == tensor[102] == tensor[110] == 1.0


def test_board_valid():
    board = LITSBoard(max_pieces_per_shape=2)
    board.play(0)  # L piece in the top left corner
    assert not board.is_valid(1)  # overlapping L piece
    assert not board.is_valid(576)  # overlapping I piece
    assert board.is_valid(582)  # adjacent I piece
    assert not board.is_valid(596)  # I piece that forms a 2x2 square
    assert board.is_valid(726)  # adjacent T piece
    assert not board.is_valid(727)  # T piece that forms a 2x2 square
    assert not board.is_valid(575)  # non-adjacent L piece
    assert not board.is_valid(1291)  # non-adjacent S piece
    board.play(582)  # adjacent I piece
    assert not board.is_valid(600)  # I piece touching the previous I piece
    assert board.is_valid(598)  # I piece not touching the previous I piece
    assert not board.is_valid(
        1044
    )  # S piece making a 2x2 square with both played pieces
    board.play(598)
    assert not board.is_valid(611)  # 3rd I piece, of which no more are available

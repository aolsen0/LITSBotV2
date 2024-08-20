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

    large_board = LITSBoard(board_size=15)
    large_tensor = large_board.to_tensor()
    for i in range(225):
        assert large_tensor[i] + large_tensor[224 - i] == 0.0


def test_board_play():
    board = LITSBoard()
    board.play(0)
    tensor = board.to_tensor()
    assert tensor[100] == tensor[101] == tensor[102] == tensor[110] == 1.0

    large_board = LITSBoard(board_size=15)
    large_board.play(1456)
    large_tensor = large_board.to_tensor()
    assert (
        large_tensor[450]
        == large_tensor[451]
        == large_tensor[452]
        == large_tensor[453]
        == 1.0
    )


def test_board_is_valid():
    board = LITSBoard(board_size=15, max_pieces_per_shape=2)
    board.play(0)  # L piece in the top left corner
    assert not board.is_valid(0)  # the same piece
    assert not board.is_valid(1)  # overlapping L piece
    assert not board.is_valid(1456)  # overlapping I piece
    assert board.is_valid(1462)  # adjacent I piece
    assert not board.is_valid(1486)  # I piece that forms a 2x2 square
    assert board.is_valid(1826)  # adjacent T piece
    assert not board.is_valid(1827)  # T piece that forms a 2x2 square
    assert not board.is_valid(1455)  # non-adjacent L piece
    assert not board.is_valid(3271)  # non-adjacent S piece
    board.play(1462)  # adjacent I piece
    assert not board.is_valid(1490)  # I piece touching the previous I piece
    assert board.is_valid(1488)  # I piece not touching the previous I piece
    assert not board.is_valid(
        2604
    )  # S piece making a 2x2 square with both played pieces
    board.play(1488)
    assert not board.is_valid(1511)  # 3rd I piece, of which no more are available


def test_board_valid_moves():
    board = LITSBoard(max_pieces_per_shape=1)
    assert len(board.valid_moves()) == 1292
    board.play(0)
    # 0 L, 6 I, 8 T, 8 S
    assert len(board.valid_moves()) == 22
    board.play(582)
    # 0 L, 0 I, 20 T, 16 S
    assert len(board.valid_moves()) == 36


def test_board_str():
    board = LITSBoard(board_size=15)
    board_str = str(board)
    assert board_str.count("X") == 30
    assert board_str.count("\n") == 14
    board.play(0)
    board_str = str(board)
    assert board_str.count("■") == 4

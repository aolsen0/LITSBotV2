from src.piece_utils import (
    PieceType,
    build_piece_list,
    get_piece_tensor,
    get_piece_type,
    get_piece_type_of_id,
)


def test_get_piece_type():
    L_cells = {(1, 2), (1, 3), (2, 3), (3, 3)}
    I_cells = {(4, 3), (5, 3), (6, 3), (7, 3)}
    T_cells = {(2, 8), (2, 9), (2, 7), (3, 8)}
    S_cells = {(9, 1), (8, 1), (8, 0), (7, 0)}
    invalid_cells = {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert get_piece_type(L_cells) == PieceType.L
    assert get_piece_type(I_cells) == PieceType.I
    assert get_piece_type(T_cells) == PieceType.T
    assert get_piece_type(S_cells) == PieceType.S
    assert get_piece_type(invalid_cells) == PieceType.Invalid


def test_build_piece_list():
    pieces = build_piece_list()
    assert len(pieces) == 1292
    assert len(set(pieces)) == 1292  # uniqueness
    for piece in pieces:
        assert get_piece_type(piece) != PieceType.Invalid

    large_pieces = build_piece_list(20)
    assert len(large_pieces) == len(set(large_pieces)) == 6152
    for piece in large_pieces:
        assert get_piece_type(piece) != PieceType.Invalid


def test_get_piece_type_from_id():
    pieces = build_piece_list()
    for i in range(1292):
        assert get_piece_type_of_id(i) == get_piece_type(pieces[i])

    large_pieces = build_piece_list(20)
    for i in range(6152):
        assert get_piece_type_of_id(i, 20) == get_piece_type(large_pieces[i])


def test_get_piece_tensor():
    a = get_piece_tensor(0, 4)
    assert a.shape == (5, 4, 4)
    assert a.sum() == 4.0
    assert a[1, 0, 0] == 1.0
    assert a[1, 0, 1] == 1.0
    assert a[1, 1, 0] == 1.0
    assert a[1, 0, 2] == 1.0

    b = get_piece_tensor(51, 4)
    assert b.sum() == 4.0
    assert b[2, 0, 2] == 1.0
    assert b[2, 1, 2] == 1.0
    assert b[2, 2, 2] == 1.0
    assert b[2, 3, 2] == 1.0

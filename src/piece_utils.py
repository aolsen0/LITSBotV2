from collections.abc import Collection
import enum
import functools
import itertools

import torch


class PieceType(enum.Enum):
    L = 1
    I = 2
    T = 3
    S = 4
    Invalid = 5

    @staticmethod
    def all_values() -> list["PieceType"]:
        return [PieceType.L, PieceType.I, PieceType.T, PieceType.S]


def taxi_distance(cell1: tuple[int, int], cell2: tuple[int, int]) -> int:
    """
    The taxicab distance between two cell locations as tuples of ints.
    """
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])


def squared_distance(cell1: tuple[int, int], cell2: tuple[int, int]) -> int:
    """
    The squared cartesian distance between two cell locations ass tuples of ints.
    """
    return (cell1[0] - cell2[0]) ** 2 + (cell1[1] - cell2[1]) ** 2


def get_piece_type(cells: Collection[tuple[int, int]]) -> PieceType:
    """
    Given a tuple of cells representing a single piece, returns which type of piece.

    Args:
        cells: Collection of 4 pairs of integers, each representing a single cell.
    Returns:
        PieceType represented by the 4 cells, and PieceType.Invalid if they do not form
        a single piece.
    """
    # We can determine the piece type by looking only at the pairwise euclidean
    # distances. It is left as an exercise for the reader that no non-pieces can have
    # these pairwise distances.
    if len(cells) != 4:
        raise ValueError("Can only check the piece type of a collection of 4 cells")
    distances = []
    for cell1 in cells:
        for cell2 in cells:
            distances.append(squared_distance(cell1, cell2))
    distances.sort()
    if distances == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 5, 5]:
        return PieceType.L
    if distances == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 9, 9]:
        return PieceType.I
    if distances == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4]:
        return PieceType.T
    if distances == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 5, 5]:
        return PieceType.S
    return PieceType.Invalid


def get_piece_type_of_id(piece_id: int, board_size: int = 10) -> PieceType:
    """
    Returns the type of the piece with the given id on the given board size.
    """
    num_L = 8 * (board_size - 1) * (board_size - 2)
    num_I = 2 * (board_size - 3) * board_size
    num_T = 4 * (board_size - 1) * (board_size - 2)
    num_S = num_T
    if piece_id < 0:
        return PieceType.Invalid
    if piece_id < num_L:
        return PieceType.L
    if piece_id < num_L + num_I:
        return PieceType.I
    if piece_id < num_L + num_I + num_T:
        return PieceType.T
    if piece_id < num_L + num_I + num_T + num_S:
        return PieceType.S
    return PieceType.Invalid


@functools.cache
def build_piece_list(board_size: int = 10) -> tuple[tuple[tuple[int, int], ...]]:
    """
    Constructs a canonical list of valid pieces and their cells.

    Pieces should almost always be referred to by their index within this list and not
    their underlying cells for the sake of efficiency.

    There are 1292 total pieces on a 10x10 board:
     - 576 L pieces. These have 4 rotations, 2 reflections, and 72 placements for each
        orientation.
     - 140 I pieces. These have 2 rotations and 70 placements each.
     - 288 T pieces. These have 4 rotations and 72 placements each.
     - 288 S pieces. These have 2 rotations, 2 reflections and 72 placements each.

    For a board with size n, there are:
     - 8 * (n - 1) * (n - 2) L pieces
     - 2 * (n - 3) * n       I pieces
     - 4 * (n - 1) * (n - 2) T pieces
     - 4 * (n - 1) * (n - 2) S pieces

    Args:
        board_size: The size of the board to generate pieces for.

    Returns:
        Tuple of the valid pieces, sorted by their type and then by location on
        the board. Each piece is a tuple of four cells of the form (row, col).
    """
    cells = list(itertools.product(range(board_size), range(board_size)))
    L_pieces = []
    I_pieces = []
    T_pieces = []
    S_pieces = []
    for cell1 in cells:
        for cell2 in cells:
            if cell1 >= cell2:
                continue
            # no two cells greater than this distance apart can be in the same piece
            if taxi_distance(cell1, cell2) > 3:
                continue
            for cell3 in cells:
                if cell2 >= cell3:
                    continue
                if taxi_distance(cell1, cell3) > 3 or taxi_distance(cell2, cell3) > 3:
                    continue
                for cell4 in cells:
                    if cell3 >= cell4:
                        continue
                    if (
                        taxi_distance(cell1, cell4) > 3
                        or taxi_distance(cell2, cell4) > 3
                        or taxi_distance(cell3, cell4) > 3
                    ):
                        continue
                    piece = (cell1, cell2, cell3, cell4)
                    piece_type = get_piece_type(piece)
                    match piece_type:
                        case PieceType.L:
                            L_pieces.append(piece)
                        case PieceType.I:
                            I_pieces.append(piece)
                        case PieceType.T:
                            T_pieces.append(piece)
                        case PieceType.S:
                            S_pieces.append(piece)
    return (*L_pieces, *I_pieces, *T_pieces, *S_pieces)


@functools.cache
def map_cells_to_id(board_size: int = 10) -> dict[tuple[tuple[int, int]], int]:
    """
    Returns a mapping from the cells of each piece to the id of that piece.
    """
    piece_list = build_piece_list(board_size)
    return {piece: i for i, piece in enumerate(piece_list)}


@functools.cache
def get_piece_tensor(piece_id: int, board_size: int = 10) -> torch.Tensor:
    """
    Returns a tensor representing the given piece.
    """
    piece_type = get_piece_type_of_id(piece_id, board_size)
    piece_list = build_piece_list(board_size)
    piece = piece_list[piece_id]
    tensor = torch.zeros(5, board_size, board_size)
    for row, col in piece:
        tensor[piece_type.value, row, col] = 1.0
    return tensor

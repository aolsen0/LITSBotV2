import collections
import itertools
import random
import torch
import torch.nn.functional as F

from src import piece_utils


class LITSBoard:
    """Represents the board state of a game of Battle of LITS."""

    def __init__(
        self, board_size: int = 10, num_xs: int = 30, max_pieces_per_shape: int = 5
    ):
        """Initializes board state with the given parameters.

        Args:
            board_size: Number of rows and columns in the board.
            num_xs: Number (each) of Xs and Os to scatter on the board. They are
                situated such that each X is opposite an O on the board.
            max_pieces_per_shape: Maximum number of pieces of each shape that can be
                placed on the board.
        """
        if 2 * num_xs > board_size**2:
            raise ValueError("cannot fit that many symbols on the board")
        self.played_ids = []
        self.board_size = board_size
        self.num_xs = num_xs
        self.max_pieces_per_shape = max_pieces_per_shape
        board_tensor = torch.zeros(board_size, board_size)
        for _ in range(num_xs):
            while True:
                row = random.randrange(board_size)
                col = random.randrange(board_size)
                if board_tensor[row, col] != 0:
                    continue
                # you cant have a symbol in the middle of an odd-length board
                if row == col == (board_size - 1) / 2:
                    continue
                board_tensor[row, col] = 1.0
                board_tensor[board_size - 1 - row, board_size - 1 - col] = -1.0
                break
        self._tensor = F.pad(
            board_tensor.reshape(board_size**2), (0, 4 * board_size**2 + 1)
        )

    def play(self, piece_id: int) -> None:
        """Play a piece on the current board.

        The piece is assumed to be legally playable. Modifies the board instance.
        """
        self.played_ids.append(piece_id)
        piece_type = piece_utils.get_piece_type_of_id(piece_id, self.board_size)
        cells = piece_utils.build_piece_list(self.board_size)[piece_id]
        for row, col in cells:
            self._tensor[
                self.board_size**2 * piece_type.value + self.board_size * row + col
            ] = 1.0

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def is_valid(self, new_piece_id: int) -> bool:
        """Check if a piece can be legally played on the current board."""
        new_piece_type = piece_utils.get_piece_type_of_id(new_piece_id, self.board_size)
        if new_piece_type == piece_utils.PieceType.Invalid:
            raise ValueError("no piece exists with that id")

        # Check if we have pieces of this type remaining
        type_counts = collections.Counter(
            piece_utils.get_piece_type_of_id(piece_id, self.board_size)
            for piece_id in self.played_ids
        )
        if type_counts[new_piece_type] >= self.max_pieces_per_shape:
            return False

        piece_list = piece_utils.build_piece_list(self.board_size)
        new_cells = piece_list[new_piece_id]
        # Check that the piece does not intersect already played pieces
        played_cells = set(
            itertools.chain(*[piece_list[played_id] for played_id in self.played_ids])
        )
        if played_cells & set(new_cells):
            return False

        # Check that the piece is adjacent to a previously played piece
        for cell in played_cells:
            for new_cell in new_cells:
                if piece_utils.taxi_distance(cell, new_cell) == 1:
                    break
            else:
                continue
            break
        else:
            return False

        # Check that the piece is not adjacent to another piece of the same type
        for played_id in self.played_ids:
            if (
                piece_utils.get_piece_type_of_id(played_id, self.board_size)
                == new_piece_type
            ):
                for new_cell in new_cells:
                    for cell in piece_list[played_id]:
                        distance = piece_utils.taxi_distance(cell, new_cell)
                        assert distance != 0
                        if distance == 1:
                            return False

        # Check that the piece does not fill a 2x2 square anywhere on the board
        total_cells = set(new_cells) | played_cells
        for cell in new_cells:
            row, col = cell
            # check each of the four possible 2x2 squares containing the cell
            for row_change in (-1, 1):
                for col_change in (-1, 1):
                    if (
                        (row + row_change, col) in total_cells
                        and (row, col + col_change) in total_cells
                        and (row + row_change, col + col_change) in total_cells
                    ):
                        return False
        return True

    def valid_moves(self) -> list[int]:
        pass

    def to_children_tensor(self) -> torch.Tensor:
        pass

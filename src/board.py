import collections
import random
import colorama
import numpy as np
import torch

from src.piece_utils import (
    PieceType,
    build_piece_list,
    get_adjacent_pieces,
    get_conflicting_pieces,
    get_flat_piece_tensor,
    get_piece_type_of_id,
    get_stacked_piece_tensor,
    get_stacked_wide_piece_tensor,
    get_total_number_of_pieces,
    taxi_distance,
)


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
        self._board_tensor = board_tensor
        self._piece_tensors = {
            piece_type: torch.zeros(board_size, board_size)
            for piece_type in PieceType.all_values()
        }
        self._to_tensor_memo = {}
        self._valid_moves_memo = {}
        self._score_change = torch.tensordot(
            board_tensor, get_stacked_piece_tensor(board_size), dims=[[0, 1], [1, 2]]
        )
        self.played_cells = set()

    def play(self, piece_id: int) -> None:
        """Play a piece on the current board.

        The piece is assumed to be legally playable. Modifies the board instance.
        """
        self.played_ids.append(piece_id)
        piece_type = get_piece_type_of_id(piece_id, self.board_size)
        cells = build_piece_list(self.board_size)[piece_id]
        self.played_cells |= set(cells)
        self._piece_tensors[piece_type] += get_flat_piece_tensor(
            piece_id, self.board_size
        )

    def _to_tensor(self, flip_xo: bool = False) -> torch.Tensor:
        if flip_xo:
            return torch.stack([-self._board_tensor, *self._piece_tensors.values()])
        return torch.stack([self._board_tensor, *self._piece_tensors.values()])

    def to_tensor(self, flip_xo: bool = False) -> torch.Tensor:
        """Return a tensor representation of the board state.

        Args:
            flip_xo: If True, flip the Xs and Os on the board.
        Returns:
            Tensor of shape (5, board_size, board_size) representing the board state.
            The first channel represents the Xs and Os on the board, and the remaining
            channels represent the cells occupied by pieces of each type."""
        key = (flip_xo, len(self.played_ids))
        if key not in self._to_tensor_memo:
            self._to_tensor_memo[key] = self._to_tensor(flip_xo)
        return self._to_tensor_memo[key]

    def is_valid(self, new_piece_id: int, fast_check: bool = False) -> bool:
        """Check if a piece can be legally played on the current board.

        Args:
            new_piece_id: ID of the piece to check.
            fast_check: If True, skip some checks that can be done more efficiently
                later. Should only be used when we know the new piece is adjacent to an
                existing piece and does not conflict with any existing pieces.
        Returns:
            True if the piece can be played, False otherwise.
        """
        new_piece_type = get_piece_type_of_id(new_piece_id, self.board_size)
        if new_piece_type == PieceType.Invalid:
            raise ValueError("no piece exists with that id")
        # Can play any move to start the game
        if len(self.played_ids) == 0:
            return True

        # Check if we have pieces of this type remaining
        type_counts = collections.Counter(
            get_piece_type_of_id(piece_id, self.board_size)
            for piece_id in self.played_ids
        )
        if type_counts[new_piece_type] >= self.max_pieces_per_shape:
            return False

        piece_list = build_piece_list(self.board_size)
        new_cells = piece_list[new_piece_id]

        if not fast_check:
            # Check that the piece does not intersect already played pieces
            if self.played_cells & set(new_cells):
                return False

            # Check that the piece is adjacent to a previously played piece
            for cell in self.played_cells:
                for new_cell in new_cells:
                    if taxi_distance(cell, new_cell) == 1:
                        break
                else:
                    continue
                break
            else:
                return False

            # Check that the piece is not adjacent to another piece of the same type
            for played_id in self.played_ids:
                if get_piece_type_of_id(played_id, self.board_size) == new_piece_type:
                    for new_cell in new_cells:
                        for cell in piece_list[played_id]:
                            distance = taxi_distance(cell, new_cell)
                            assert distance != 0
                            if distance == 1:
                                return False

        # Check that the piece does not fill a 2x2 square anywhere on the board
        total_cells = set(new_cells) | self.played_cells
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
        """Return a list of all piece ids which can currently be played."""
        if len(self.played_ids) == 0:
            return list(range(get_total_number_of_pieces(self.board_size)))
        if len(self.played_ids) in self._valid_moves_memo:
            return self._valid_moves_memo[len(self.played_ids)]
        if len(self.played_ids) == 1:
            result = sorted(get_adjacent_pieces(self.board_size)[self.played_ids[0]])
            self._valid_moves_memo[1] = result
            return result
        if len(self.played_ids) - 1 in self._valid_moves_memo:
            # If we have already computed the valid moves for the previous state, we can
            # quickly compute the valid moves for the current state by considering only
            # the pieces adjacent to the last played piece and the pieces that were
            # valid in the previous state.
            candidates = set(self._valid_moves_memo[len(self.played_ids) - 1])
            candidates |= get_adjacent_pieces(self.board_size)[self.played_ids[-1]]
            fast_check = True
        else:
            candidates = set(range(get_total_number_of_pieces(self.board_size)))
            fast_check = False
        for played_id in self.played_ids:
            candidates -= get_conflicting_pieces(self.board_size)[played_id]
        legal = []
        for i in candidates:
            if self.is_valid(i, fast_check):
                legal.append(i)
        self._valid_moves_memo[len(self.played_ids)] = legal
        return legal

    @staticmethod
    def subsequent_valid_static(
        board_size: int,
        max_pieces_per_shape: int,
        previous_legal: list[int],
        played_ids: list[int],
        played_cells: set[tuple[int, int]],
    ) -> list[int]:
        """Given the list of legal moves for the previous state, return the list of
        legal moves for the current state.

        This is largely a static version of `valid_moves`, as the previously played
        pieces are sufficient to determine the legal moves for the current state.

        Args:
            board_size: Number of rows and columns in the board.
            max_pieces_per_shape: Maximum number of pieces of each shape that can be
                placed on the board.
            previous_legal: List of legal moves for the previous state.
            played_ids: List of piece ids that have been played.
            played_cells: Set of cells that have been played on.
        Returns:
            List of legal moves for the current state.
        """
        if len(played_ids) == 0:
            return list(range(get_total_number_of_pieces(board_size)))
        if len(played_ids) == 1:
            candidates = set()
        else:
            candidates = set(previous_legal)
        candidates |= get_adjacent_pieces(board_size)[played_ids[-1]]
        for played_id in played_ids:
            candidates -= get_conflicting_pieces(board_size)[played_id]
        type_counts = collections.Counter(
            get_piece_type_of_id(piece_id, board_size) for piece_id in played_ids
        )

        def check_legal(piece_id: int) -> bool:
            new_piece_type = get_piece_type_of_id(piece_id, board_size)
            if type_counts[new_piece_type] >= max_pieces_per_shape:
                return False
            new_cells = build_piece_list(board_size)[piece_id]
            total_cells = set(new_cells) | played_cells
            for cell in new_cells:
                row, col = cell
                for row_change in (-1, 1):
                    for col_change in (-1, 1):
                        if (
                            (row + row_change, col) in total_cells
                            and (row, col + col_change) in total_cells
                            and (row + row_change, col + col_change) in total_cells
                        ):
                            return False
            return True

        legal = []
        for i in candidates:
            if check_legal(i):
                legal.append(i)
        return legal

    def __str__(self) -> str:
        board_str = np.full(
            [self.board_size, self.board_size], " ", dtype=np.dtypes.StringDType
        )
        board_str[self._board_tensor > 0] = "X"
        board_str[self._board_tensor < 0] = "O"

        board_str[self._piece_tensors[PieceType.L] > 0] = (
            f"{colorama.Fore.RED}■{colorama.Style.RESET_ALL}"
        )
        board_str[self._piece_tensors[PieceType.I] > 0] = (
            f"{colorama.Fore.YELLOW}■{colorama.Style.RESET_ALL}"
        )
        board_str[self._piece_tensors[PieceType.T] > 0] = (
            f"{colorama.Fore.GREEN}■{colorama.Style.RESET_ALL}"
        )
        board_str[self._piece_tensors[PieceType.S] > 0] = (
            f"{colorama.Fore.CYAN}■{colorama.Style.RESET_ALL}"
        )
        header = [
            "     " + " ".join(chr(i + ord("A")) for i in range(self.board_size)),
            "    " + "–" * (2 * self.board_size + 1),
        ]
        rows = [
            f"{str(i + 1).rjust(2)} | " + " ".join(row) + " |"
            for i, row in enumerate(board_str)
        ]
        footer = header[1:]
        return ("\n").join(header + rows + footer)

    def to_children_tensor(
        self, pieces_to_use: list[int], flip_xo: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the tensor representing all possible board states after the given
        pieces are played. Also return the change in score that would result from each
        piece being played.

        Args:
            pieces_to_use: List of piece ids to consider playing.
        Returns:
            - Tensor of shape (len(pieces_to_use), 5, board_size, board_size)
                representing the board states after each piece is played.
            - Tensor of shape (len(pieces_to_use),) representing the change in score
                that would result from each piece being played.
        """
        current_tensor = self.to_tensor(flip_xo)
        piece_tensors = get_stacked_wide_piece_tensor(self.board_size)[pieces_to_use]
        score_changes = self._score_change[pieces_to_use]
        if flip_xo:
            score_changes = -score_changes
        return current_tensor + piece_tensors, score_changes

    @staticmethod
    def to_children_tensor_static(
        board_size: int,
        current_tensor: torch.Tensor,
        pieces_to_use: list[int],
    ) -> torch.Tensor:
        """Return the tensor representing all possible board states after the given
        pieces are played.

        Args:
            board_size: Number of rows and columns in the board.
            current_tensor: Tensor representing the current board state.
            pieces_to_use: List of piece ids to consider playing.
        Returns:
            Tensor of shape (len(pieces_to_use), 5, board_size, board_size)
            representing the board states after each piece is played.
        """
        flipped_tensor = torch.stack([-current_tensor[0], *current_tensor[1:]])
        piece_tensors = get_stacked_wide_piece_tensor(board_size)[pieces_to_use]
        return flipped_tensor + piece_tensors

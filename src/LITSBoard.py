import random
import torch
import torch.nn.functional as F


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
                board_tensor[row, col] = 1.0
                board_tensor[board_size - 1 - row, board_size - 1 - col] = -1.0
                break
        self._tensor = F.pad(
            board_tensor.reshape(board_size**2), (0, 4 * board_size**2 + 1)
        )

    def play(self, piece_id: int) -> None:
        pass

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def is_valid(self, piece_id: int) -> bool:
        pass

    def valid_moves(self) -> list[int]:
        pass

    def to_children_tensor(self) -> torch.Tensor:
        pass

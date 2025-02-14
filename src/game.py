import random
import torch
import torch.nn as nn

from src.board import LITSBoard
from src.piece_utils import map_cells_to_id


class LITSGame:
    """Represents the full game state of a game of battle of LITS"""

    def __init__(
        self, board_size: int = 10, num_xs: int = 30, max_pieces_per_shape: int = 5
    ):
        """Initializes game state with the given parameters.

        Args:
            board_size: Number of rows and columns in the board.
            num_xs: Number (each) of Xs and Os to scatter on the board. They are
                situated such that each X is opposite an O on the board.
            max_pieces_per_shape: Maximum number of pieces of each shape that can be
                placed on the board.
        """
        self.board = LITSBoard(board_size, num_xs, max_pieces_per_shape)
        self.board_size = board_size
        self.current_player = 0
        self.swapped = False
        self.completed = False

    def play(self, piece_id: int) -> None:
        """Make a move in the game.

        If it is the second player's first turn, they can choose to swap sides and
        should not play a piece. Passing -1 as piece_id will swap which player is
        X and which is O, and will not place a piece on the board. A non-negative
        piece_id will play the piece with that id on the board without swapping.

        Args:
            piece_id: The id of the piece to play, or -1 to swap sides.
        Raises:
            ValueError: If the game is already completed, or if the move is invalid.
        """
        if self.completed:
            raise ValueError("Cannot play on a completed game")
        if self.is_swappable() and piece_id == -1:
            self.swapped = True
            self.current_player = 0
            return
        if not self.board.is_valid(piece_id):
            raise ValueError("Invalid move")
        self.board.play(piece_id)
        self.current_player = 1 - self.current_player
        if not self.board.valid_moves():
            self.completed = True

    def score(self) -> float:
        """Return the current score of the game from the perspective of the first
        player.

        If the game is in progress or the players finish with differeny scores, this is
        the difference between the number of uncovered Xs and Os. If the game is
        finished and the players finish with the same score, this is ±0.5, depending on
        who played last (as the player who played last wins in case of a tie).
        """
        tensor = self.board.to_tensor()
        covered = tensor[1:].sum(axis=0)
        symbols = tensor[0].clone()
        symbols[covered > 0] = 0.0
        score = symbols.sum().item()
        if self.swapped:
            score = -score
        if self.completed and score == 0.0:
            return -0.5 if self.current_player == 0 else 0.5
        return score

    def is_swappable(self) -> bool:
        """Return whether the current player can swap sides."""
        return self.current_player == 1 and len(self.board.played_ids) == 1

    def prompt(self) -> None:
        """Receive input from the user for the next move"""
        if self.completed:
            print("Game over")
            return
        print(self.board)
        print(f"Player {self.current_player + 1} to play")
        if self.is_swappable():
            print("Do you want to swap sides? (y/n)")
            response = input()
            if response.lower() == "y":
                self.play(-1)
                return
        print("Enter the cells of the piece you want to play")
        while True:
            user_cells = input().split()
            if len(user_cells) == 8:
                cells = [
                    (int(user_cells[i + 1]) - 1, ord(user_cells[i]) - ord("A"))
                    for i in range(0, 8, 2)
                ]
                cells.sort()
                try:
                    piece_id = map_cells_to_id(self.board_size)[tuple(cells)]
                except KeyError:
                    print("Cells do not form a valid piece")
                    continue
                if not self.board.is_valid(piece_id):
                    print("Not a valid move on the current board")
                    continue
                self.play(piece_id)
                return
            print(
                "Invalid input, try again. Input should be four cells, in the format "
                "'A 1 B 1 C 1 D 1'"
            )

    def generate_examples(
        self, model: nn.Module, epsilon: float = 0.2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate examples for training a reinforcement learning model.

        Args:
            model: The model to generate examples for.
            epsilon: The probability of choosing a random move instead of the best move.
        Returns:
            A list of examples, each of the form (board_tensor, move_tensor, value).
        """
        if self.board.played_ids:
            raise ValueError("Cannot generate examples for a game in progress")

        # don't generate examples for the first move
        if random.random() < epsilon:
            piece_id = random.choice(self.board.valid_moves())
        else:
            children_tensor, score_changes = self.board.to_children_tensor(
                self.board.valid_moves()
            )
            with torch.no_grad():
                values = model(children_tensor) - score_changes.unsqueeze(1)
            piece_id = values.abs().argmin().item()
        self.play(piece_id)

        inputs = []
        outputs = []
        while not self.completed:
            flip = bool(len(self.board.played_ids) % 2)
            inputs.append(self.board.to_tensor(flip))
            moves = self.board.valid_moves()
            children_tensor, score_changes = self.board.to_children_tensor(
                moves, not flip
            )
            with torch.no_grad():
                values = model(children_tensor) - score_changes.unsqueeze(1)
            outputs.append(-values.min().item())
            if random.random() < epsilon:
                piece_id = random.choice(moves)
            else:
                piece_id = moves[values.argmin().item()]
            self.play(piece_id)

        # add the final state
        flip = bool(len(self.board.played_ids) % 2)
        inputs.append(self.board.to_tensor(flip))
        value = -self.score() if flip else self.score()
        outputs.append(1.0 if value > 0 else -1.0)

        return torch.stack(inputs), torch.tensor(outputs).unsqueeze(1)

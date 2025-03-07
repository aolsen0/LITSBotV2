import random
import torch
import torch.nn as nn

from src.board import LITSBoard
from src.piece_utils import get_total_number_of_pieces, map_cells_to_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def play(self, piece_id: int, check_validity: bool = True) -> None:
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
        if check_validity and not self.board.is_valid(piece_id):
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
        self, model: nn.Module, epsilon: float = 0.2, single_output: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate examples for training a reinforcement learning model.

        This assumes that the model is on cuda if available.

        Args:
            model: The model to generate examples for.
            epsilon: The probability of choosing a random move instead of the best move.
            single_output: Whether the model estimates the value of the current game
                state, or the value of each possible move in the current game state.
                Models estimating the value of each possible move should output a tensor
                of shape (batch_size, 2, num_moves), where the first channel contains
                the value of the game state after each move, and the second channel
                contains the estimated legality of each move.
        Returns:
            - A tensor of inputs, where each input is a tensor representing the board
                state.
            - A tensor of expected ouputs for each input, based on the best value(s)
                predicted by the model after one move.
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
            if single_output:
                with torch.no_grad():
                    values = model(
                        children_tensor.to(device)
                    ) - score_changes.unsqueeze(1).to(device)
                piece_id = values.abs().argmin().item()
            else:
                output = model(children_tensor.to(device))
                score = (
                    torch.where(output[:, 1] > 0.5, output[:, 0], -float("inf"))
                    .max(dim=1)
                    .values
                )
                piece_id = (score - score_changes.to(device)).abs().argmin().item()
        self.play(piece_id, False)

        inputs = []
        outputs = []
        while not self.completed:
            # flip so that the current player is always trying to cover Xs
            flip = bool(len(self.board.played_ids) % 2)
            inputs.append(self.board.to_tensor(flip))
            moves = self.board.valid_moves()
            children_tensor, score_changes = self.board.to_children_tensor(
                moves, not flip
            )

            if single_output:
                with torch.no_grad():
                    values = model(
                        children_tensor.to(device)
                    ) - score_changes.unsqueeze(1).to(device)
                outputs.append(-values.min(dim=0).values)
            else:
                with torch.no_grad():
                    values = model(children_tensor.to(device))
                score = (
                    torch.where(values[:, 1] > 0.5, values[:, 0], -float("inf"))
                    .max(dim=1)
                    .values
                )
                print(moves)
                print(score)
                # if there are no legal moves, than no more moves can be played and
                # the remaining score must be 0.
                score[score == -float("inf")] = 0.0
                correct = score_changes.to(device) - score

                output = torch.zeros(
                    [2, get_total_number_of_pieces(self.board_size)], device=device
                )
                output[0, moves] = correct
                output[1, moves] = 1.0
                outputs.append(output)
            if random.random() < epsilon:
                piece_id = random.choice(moves)
            else:
                if single_output:
                    piece_id = moves[values.argmin().item()]
                else:
                    piece_id = moves[correct.argmax().item()]
            self.play(piece_id, False)

        # add the final state
        flip = bool(len(self.board.played_ids) % 2)
        inputs.append(self.board.to_tensor(flip))
        if single_output:
            outputs.append(torch.zeros([1], device=device))
        else:
            outputs.append(
                torch.zeros(
                    [2, get_total_number_of_pieces(self.board_size)], device=device
                )
            )

        return torch.stack(inputs), torch.stack(outputs)

    def play_best(self, model: nn.Module, single_output: bool = True) -> None:
        """Play the best move according to the given model.

        Assumes the model estimates future score changes, as suggested by the examples
        generated by the generate_examples method.
        """
        if self.completed:
            raise ValueError("Cannot play on a completed game")
        moves = self.board.valid_moves()
        flip = self.current_player ^ self.swapped
        children_tensor, score_changes = self.board.to_children_tensor(moves, not flip)
        if single_output:
            with torch.no_grad():
                values = model(children_tensor.to(device)) - score_changes.unsqueeze(
                    1
                ).to(device)
        else:
            with torch.no_grad():
                output = model(children_tensor.to(device))
            score = (
                torch.where(output[:, 1] > 0.5, output[:, 0], -float("inf"))
                .max(axis=1)
                .values
            )
            values = score - score_changes.to(device)
        if len(self.board.played_ids) == 0:
            # make an equal move as the opponent can choose to swap sides
            piece_id = moves[values.abs().argmin().item()]
        elif self.is_swappable() and values.min().item() + self.score() > 0:
            # we think we are behind, so we swap sides
            piece_id = -1
        else:
            piece_id = moves[values.argmin().item()]
        self.play(piece_id)

    def evaluate(self, model: nn.Module, single_output: bool = True) -> float:
        """Evaluate the game state using the given model.

        Should not be expected to make sense before the second player has chosen whether
        or not to swap sides.

        Args:
            model: The model to evaluate the game state with.
        Returns:
            The value of the game state from the perspective of the first player.
        """
        flip = self.current_player ^ self.swapped
        with torch.no_grad():
            if single_output:
                value = model(self.board.to_tensor(flip).unsqueeze(0).to(device)).item()
            else:
                output = model(self.board.to_tensor(flip).unsqueeze(0).to(device))
                value = (
                    torch.where(output[:, 1] > 0.5, output[:, 0], -float("inf"))
                    .max()
                    .item()
                )
        if self.current_player:
            value = -value
        return self.score() + value

    def play_against(
        self, model: nn.Module, model_player: int = 1, single_output: bool = True
    ) -> None:
        """Play a game against the given model."""
        while not self.completed:
            if self.current_player == model_player - 1:
                self.play_best(model, single_output)
            else:
                self.prompt()
        print(self.board)
        winner = 1 if self.score() > 0 else 2
        print(f"Player {winner} wins")
        print(f"Score: {self.score()}")

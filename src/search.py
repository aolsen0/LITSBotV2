import time
from typing import Union
import torch
from src.board import LITSBoard
from src.model import LITSModel, MoveModel
from src.piece_utils import build_piece_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEPTH_0_CLIP = 1.6
DEPTH_1_CLIP = 1.2


class SearchNode:
    """Class containing functionality to run game tree searches."""

    def __init__(
        self,
        board_size: int,
        max_pieces_per_shape: int,
        score_changes: torch.Tensor,
        parent: Union["SearchNode", None],
        model: LITSModel | MoveModel,
        single_output: bool,
        played_pieces: list[int],
        played_cells: set[tuple[int, int]],
        curr_tensor: torch.Tensor,
        curr_output: torch.Tensor | None,
        skip_legality_check: bool = False,
        legal_moves: list[int] | None = None,
    ):
        self.board_size = board_size
        self.max_pieces_per_shape = max_pieces_per_shape
        self.score_changes = score_changes
        self.parent = parent
        self.children: list["SearchNode" | None] = []
        self.model = model
        self.single_output = single_output
        self.played_pieces = played_pieces
        self.played_cells = played_cells
        self.skip_legality_check = skip_legality_check
        if single_output and skip_legality_check:
            raise ValueError(
                "Models with single output do not provide legality information."
            )
        if parent is None:
            if legal_moves is None:
                raise ValueError("Legal moves must be provided for the root node.")
            self.legal_moves = legal_moves
        else:
            if skip_legality_check:
                self.legal_moves = (curr_output[1] > 0.5).nonzero().squeeze().tolist()
            else:
                self.legal_moves = LITSBoard.subsequent_valid_static(
                    board_size,
                    max_pieces_per_shape,
                    parent.legal_moves,
                    played_pieces,
                    played_cells,
                )
        if not self.legal_moves:
            self.value = 0.0
            return
        self.children_tensor = LITSBoard.to_children_tensor_static(
            board_size, curr_tensor, self.legal_moves
        )
        self.important_score_changes = self.score_changes[self.legal_moves]
        if len(played_pieces) % 2:
            self.important_score_changes = -self.important_score_changes
        with torch.no_grad():
            self.children_output = model(self.children_tensor.to(device)).to("cpu")
        if single_output:
            value = self.children_output.reshape(-1) + self.important_score_changes
        else:
            score = (
                torch.where(
                    self.children_output[:, 1] > 0.5,
                    self.children_output[:, 0],
                    -float("inf"),
                )
                .max(dim=1)
                .values
            )
            score[score == -float("inf")] = 0.0
            value = score + self.important_score_changes
        self.all_values = value
        if self.played_pieces:
            self.value = -value.min().item()
            self.best_move_index = value.argmin().item()
        else:
            self.value = -value.abs().min().item()
            self.best_move_index = value.abs().argmin().item()

    def add_children(self) -> None:
        """Add children to the current node."""
        if self.children:
            return
        for i in range(self.children_tensor.shape[0]):
            curr_output = self.children_output[i] if self.skip_legality_check else None
            curr_value = self.all_values[i].item()
            if (self.played_pieces and -curr_value < self.value - DEPTH_0_CLIP) or (
                not self.played_pieces and -abs(curr_value) < self.value - DEPTH_0_CLIP
            ):
                self.children.append(None)
                continue
            piece_id = self.legal_moves[i]
            played_pieces = self.played_pieces + [piece_id]
            played_cells = self.played_cells | set(
                build_piece_list(self.board_size)[piece_id]
            )
            curr_tensor = self.children_tensor[i]
            child_node = SearchNode(
                self.board_size,
                self.max_pieces_per_shape,
                self.score_changes,
                self,
                self.model,
                self.single_output,
                played_pieces,
                played_cells,
                curr_tensor,
                curr_output,
                self.skip_legality_check,
            )
            self.children.append(child_node)

    def alpha_beta_search(
        self,
        depth: int,
        alpha: float = -float("inf"),
        beta: float = float("inf"),
        alpha_player: bool = True,
        kill_time: float | None = None,
    ) -> float:
        """
        Perform alpha-beta pruning search to the specified depth.

        Args:
            depth: Maximum depth to search
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: Whether current player is maximizing

        Returns:
            Tuple of (best_value, best_move_index) where best_move_index is the index
            in self.legal_moves corresponding to the best move
        """
        if depth == 0 or not self.legal_moves:
            return self.value

        # Create children if they don't exist yet
        if not self.children:
            self.add_children()

        # Get children with their evaluation scores for sorting
        child_values = [
            (i, child.value + self.important_score_changes[i].item())
            for i, child in enumerate(self.children)
            if child is not None
        ]
        if self.played_pieces:
            child_values.sort(key=lambda x: x[1])
        else:
            child_values.sort(key=lambda x: abs(x[1]))
        best_prior_value = child_values[0][1]
        for i, value in child_values:
            # these cases are probably never optimal moves, it's fine to delete them
            if (self.played_pieces and value > best_prior_value + DEPTH_1_CLIP) or (
                not self.played_pieces
                and abs(value) > abs(best_prior_value) + DEPTH_1_CLIP
            ):
                self.children[i] = None

        best_value = -float("inf")
        best_move_index = None

        for i, value in child_values:
            child = self.children[i]
            if child is None:
                continue

            # Recursively search child node
            next_score = self.important_score_changes[i].item()
            child_value = (
                child.alpha_beta_search(
                    depth - 1,
                    alpha + next_score,
                    beta + next_score,
                    not alpha_player,
                    kill_time,
                )
                + next_score
            )
            if not self.played_pieces:
                child_value = abs(child_value)

            if -child_value > best_value:
                best_value = -child_value
                best_move_index = i

                if alpha_player:
                    alpha = max(alpha, best_value)
                    if not self.played_pieces:
                        beta = min(beta, -best_value)
                else:
                    beta = min(beta, -best_value)

                # Prune if possible
                if beta <= alpha:
                    break

            if kill_time is not None and time.time() >= kill_time:
                break

        self.value = best_value
        self.best_move_index = best_move_index

        return best_value

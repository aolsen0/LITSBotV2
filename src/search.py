from typing import Union
import torch
from src.board import LITSBoard
from src.model import LITSModel, MoveModel
from src.piece_utils import build_piece_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.children = []
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
            self.children_output = model(self.children_tensor.to(device))
        if single_output:
            value = self.children_output.reshape(-1) + self.important_score_changes.to(
                device
            )
            self.value = -value.min().item()
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
            value = score + self.important_score_changes.to(device)
            print(value)
            self.value = -value.min().item()

    def add_children(self):
        """Add children to the current node."""
        for i in range(self.children_tensor.shape[0]):
            piece_id = self.legal_moves[i]
            played_pieces = self.played_pieces + [piece_id]
            played_cells = self.played_cells | set(
                build_piece_list(self.board_size)[piece_id]
            )
            curr_tensor = self.children_tensor[i]
            curr_output = self.children_output[i]
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

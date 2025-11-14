import copy
from unittest.mock import patch
import torch
from src.board import LITSBoard
from src.piece_utils import get_stacked_piece_tensor
from src.search import SearchNode


def test_searchnode_init():
    board = LITSBoard(board_size=4, num_xs=4)

    real_board = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    board._board_tensor = real_board
    board._score_change = torch.tensordot(
        board._board_tensor, get_stacked_piece_tensor(4), dims=[[0, 1], [1, 2]]
    )
    board.play(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model(tensor):
        return -5 * torch.tensor([list(range(tensor.shape[0]))]).to(device).T

    parent_node = SearchNode(
        board.board_size,
        board.max_pieces_per_shape,
        board._score_change,
        None,
        model,
        True,
        board.played_ids,
        board.played_cells,
        board.to_tensor(flip_xo=True),
        None,
        legal_moves=board.valid_moves(),
    )
    assert parent_node.value == 47.0

    board.play(52)
    child = SearchNode(
        board.board_size,
        board.max_pieces_per_shape,
        board._score_change,
        parent_node,
        model,
        True,
        board.played_ids,
        board.played_cells,
        board.to_tensor(flip_xo=False),
        None,
    )
    assert child.value == 13.0

    def other_model(tensor):
        return -torch.tensor(list(range(832))).to(device).reshape(4, 2, 104) / 100 + 4

    output = torch.zeros(2, 104)
    output[1, [76, 78, 100, 101]] = 1.0

    child = SearchNode(
        board.board_size,
        board.max_pieces_per_shape,
        board._score_change,
        parent_node,
        other_model,
        False,
        board.played_ids,
        board.played_cells,
        board.to_tensor(flip_xo=False),
        output,
        skip_legality_check=True,
    )
    assert child.legal_moves == [76, 78, 100, 101]
    assert child.value == -2.0


@patch("src.search.DEPTH_0_CLIP", 100.0)
def test_searchnode_create_children():
    board = LITSBoard(board_size=4, num_xs=4)
    real_board = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    board._board_tensor = real_board
    board._score_change = torch.tensordot(
        board._board_tensor, get_stacked_piece_tensor(4), dims=[[0, 1], [1, 2]]
    )
    board.play(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model(tensor):
        return -5 * torch.tensor([list(range(tensor.shape[0]))]).to(device).T

    parent_node = SearchNode(
        board.board_size,
        board.max_pieces_per_shape,
        board._score_change,
        None,
        model,
        True,
        board.played_ids,
        board.played_cells,
        board.to_tensor(flip_xo=True),
        None,
        legal_moves=board.valid_moves(),
    )
    parent_node.add_children()
    assert len(parent_node.children) == 10
    child = parent_node.children[0]
    assert child.value == 13.0
    assert child.played_pieces == [0, 52]


@patch("src.search.DEPTH_0_CLIP", 100.0)
@patch("src.search.DEPTH_1_CLIP", 100.0)
def test_searchnode_alpha_beta():
    board = LITSBoard(board_size=5, num_xs=6)
    real_board = torch.tensor(
        [
            [0.0, -1.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, -1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
        ]
    )
    board._board_tensor = real_board
    board._score_change = torch.tensordot(
        board._board_tensor, get_stacked_piece_tensor(5), dims=[[0, 1], [1, 2]]
    )

    def model(tensor):
        print(tensor.shape)
        return torch.zeros(tensor.shape[0])

    root = SearchNode(
        board.board_size,
        board.max_pieces_per_shape,
        board._score_change,
        None,
        model,
        True,
        board.played_ids,
        board.played_cells,
        board.to_tensor(flip_xo=True),
        None,
        legal_moves=[101, 110],
    )
    assert root.alpha_beta_search(0) == 0.0
    assert root.best_move_index == 1
    assert root.alpha_beta_search(1) == -1.0
    assert root.best_move_index == 0
    board.play(0)

    original_alpha_beta_search = copy.deepcopy(SearchNode.alpha_beta_search)
    all_args = []

    def arg_aggregator(*args):
        all_args.append(args)
        return original_alpha_beta_search(*args)

    with patch.object(SearchNode, "alpha_beta_search", arg_aggregator):
        root = SearchNode(
            board.board_size,
            board.max_pieces_per_shape,
            board._score_change,
            None,
            model,
            True,
            board.played_ids,
            board.played_cells,
            board.to_tensor(flip_xo=True),
            None,
            legal_moves=[101, 110],
        )
        assert root.alpha_beta_search(0) == 0.0
        assert root.alpha_beta_search(5) == 0.0

    wrong_paths = 0
    for args in all_args:
        if args[0].played_pieces[:2] == [0, 101]:
            wrong_paths += 1
    # 101 is a much worse first move, we should realize so after testing the first
    # possible response and prune
    assert wrong_paths <= 2

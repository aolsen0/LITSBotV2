import tempfile
import shutil

import torch

from src.board import LITSBoard
from src.model import LITSModel, MoveModel


def test_model():
    model = LITSModel(
        name="test",
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=2,
        num_linear_layers=2,
    )
    board = LITSBoard(board_size=12)
    tensor, _ = board.to_children_tensor(list(range(32)))
    output = model(tensor)
    assert output.shape == (32, 1)

    model = LITSModel(
        name="test",
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=2,
        num_linear_layers=1,
    )
    output = model(tensor)
    assert output.shape == (32, 1)

    model = LITSModel(
        name="test",
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=0,
        num_linear_layers=2,
    )
    output = model(tensor)
    assert output.shape == (32, 1)


def test_move_model():
    model = MoveModel(
        name="test",
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=2,
        num_linear_layers=2,
    )
    board = LITSBoard(board_size=12)
    tensor, _ = board.to_children_tensor(list(range(32)))
    output = model(tensor)
    assert output.shape == (32, 2, 1976)

    model = MoveModel(
        name="test",
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=2,
        num_linear_layers=1,
    )
    output = model(tensor)
    assert output.shape == (32, 2, 1976)

    model = MoveModel(
        name="test",
        board_size=12,
        num_xs=45,
        max_pieces_per_shape=8,
        num_conv_layers=0,
        num_linear_layers=2,
    )
    output = model(tensor)
    assert output.shape == (32, 2, 1976)


def test_save_load_lits_model():
    """Test that LITSModel can be saved and loaded correctly."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = LITSModel(
            name="test_save_load",
            board_size=12,
            num_xs=45,
            max_pieces_per_shape=8,
            num_conv_layers=2,
            num_linear_layers=2,
        )
        model.save(identifier="test_checkpoint", base_dir=temp_dir)
        board = LITSBoard(board_size=12)
        tensor, _ = board.to_children_tensor(list(range(32)))
        with torch.no_grad():
            original_output = model(tensor)
        loaded_model = LITSModel.load(
            name="test_save_load", identifier="test_checkpoint", base_dir=temp_dir
        )
        with torch.no_grad():
            loaded_output = loaded_model(tensor)

        assert torch.allclose(original_output, loaded_output, atol=1e-6)
        assert loaded_model.name == model.name
        assert loaded_model.board_size == model.board_size
        assert loaded_model.num_xs == model.num_xs
        assert loaded_model.max_pieces_per_shape == model.max_pieces_per_shape
        assert loaded_model.single_output == model.single_output
    finally:
        shutil.rmtree(temp_dir)


def test_save_load_move_model():
    """Test that MoveModel can be saved and loaded correctly."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = MoveModel(
            name="test_save_load_move",
            board_size=12,
            num_xs=45,
            max_pieces_per_shape=8,
            num_conv_layers=2,
            num_linear_layers=2,
        )
        model.save(identifier="test_checkpoint", base_dir=temp_dir)
        board = LITSBoard(board_size=12)
        tensor, _ = board.to_children_tensor(list(range(32)))
        with torch.no_grad():
            original_output = model(tensor)
        loaded_model = MoveModel.load(
            name="test_save_load_move", identifier="test_checkpoint", base_dir=temp_dir
        )
        with torch.no_grad():
            loaded_output = loaded_model(tensor)
        assert torch.allclose(original_output, loaded_output, atol=1e-6)

        assert loaded_model.name == model.name
        assert loaded_model.board_size == model.board_size
        assert loaded_model.num_xs == model.num_xs
        assert loaded_model.max_pieces_per_shape == model.max_pieces_per_shape
        assert loaded_model.single_output == model.single_output
    finally:
        shutil.rmtree(temp_dir)

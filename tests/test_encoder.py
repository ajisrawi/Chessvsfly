import os
import sys

import chess
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flybrain.encoder import N_FEATURES, N_SQUARE_FEATURES, board_features  # noqa: E402

CIRCUIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "circuit.npz")


def test_feature_layout():
    x = board_features(chess.Board())
    assert x.shape == (N_FEATURES,)
    assert x.sum() == 32 + 4  # 32 pieces + four castling rights
    # white pawn on a2 -> square index 8, channel 0
    assert x[8 * 12 + 0] == 1
    # black king on e8 -> square 60, channel 6+5
    assert x[60 * 12 + 11] == 1


def test_perspective_is_colour_symmetric():
    """Mirroring the board and swapping colours must give the identical code."""
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    mirrored = b.mirror()  # swaps colours and flips ranks; side to move flips too
    assert np.array_equal(board_features(b), board_features(mirrored))


def test_en_passant_and_castling_flags():
    b = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2")
    x = board_features(b)
    assert x[N_SQUARE_FEATURES + 4] == 1
    b2 = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b - - 0 2")
    assert board_features(b2)[N_SQUARE_FEATURES:N_SQUARE_FEATURES + 4].sum() == 0


@pytest.mark.skipif(not os.path.exists(CIRCUIT), reason="circuit not built")
def test_kc_code_is_sparse_and_deterministic():
    from flybrain.connectome import load_circuit
    from flybrain.encoder import FlyEncoder

    enc = FlyEncoder(load_circuit(CIRCUIT))
    boards = [chess.Board(), chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")]
    K1 = enc.encode(boards)
    K2 = enc.encode(boards)
    assert np.array_equal(K1, K2)
    density = (K1 > 0).mean(axis=1)
    assert np.all(density > 0.005) and np.all(density < 0.3)
    assert enc.W_in.shape == (4064, N_FEATURES)

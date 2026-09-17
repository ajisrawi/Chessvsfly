"""Board -> projection neurons -> Kenyon cells.

This is the *fixed* (non-plastic) part of the fly brain.  A chess position is
presented to the fly by activating projection neurons (PNs) that innervate the
mushroom-body calyx, one PN per board feature.  The connectome's PN->KC
wiring then produces a sparse Kenyon-cell (KC) code, sharpened by the GABAergic
APL feedback neuron exactly as in the real fly (Lin et al. 2014) and coloured by
the massive KC->KC recurrence found in the connectome.

Everything here is deterministic and is mirrored bit-for-bit by the JavaScript
implementation in ``web/flybrain.js``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import chess

N_SQUARE_FEATURES = 64 * 12          # square x (own P,N,B,R,Q,K, enemy P,N,B,R,Q,K)
N_EXTRA = 5                          # own K/Q castling, enemy K/Q castling, en passant
N_FEATURES = N_SQUARE_FEATURES + N_EXTRA   # 773

PIECE_INDEX = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


def board_features(board: chess.Board) -> np.ndarray:
    """Binary feature vector (773,) from the perspective of the side to move.

    The board is mirrored so that the side to move always plays "up the board";
    this lets one set of synapses serve both colours.
    """
    x = np.zeros(N_FEATURES, dtype=np.float32)
    us = board.turn
    flip = us == chess.BLACK
    for sq, piece in board.piece_map().items():
        s = chess.square_mirror(sq) if flip else sq
        ch = PIECE_INDEX[piece.piece_type] + (0 if piece.color == us else 6)
        x[s * 12 + ch] = 1.0
    base = N_SQUARE_FEATURES
    x[base + 0] = float(board.has_kingside_castling_rights(us))
    x[base + 1] = float(board.has_queenside_castling_rights(us))
    x[base + 2] = float(board.has_kingside_castling_rights(not us))
    x[base + 3] = float(board.has_queenside_castling_rights(not us))
    x[base + 4] = float(board.ep_square is not None)
    return x


def _row_normalize(m: sp.csr_matrix) -> sp.csr_matrix:
    m = m.tocsr().astype(np.float32)
    s = np.asarray(m.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    return sp.diags(1.0 / s).dot(m).tocsr().astype(np.float32)


def _col_normalize(m: sp.csr_matrix) -> sp.csr_matrix:
    """Each presynaptic neuron distributes one unit of output over its synapses."""
    m = m.tocsc().astype(np.float32)
    s = np.asarray(m.sum(axis=0)).ravel()
    s[s == 0] = 1.0
    return m.dot(sp.diags(1.0 / s)).tocsr().astype(np.float32)


def default_feature_map(circuit: dict, feature_frequency: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Assign each board feature to one projection neuron.

    Projection neurons are ranked by how many Kenyon cells they contact; the
    most frequently occurring board features get the best-connected neurons
    (a feature seen by a neuron with two Kenyon-cell partners is nearly
    invisible).  Without frequencies the assignment is a seeded permutation of
    the 773 best-connected inputs.
    """
    partners = np.asarray((circuit["W_in"] > 0).sum(axis=0)).ravel()
    pn_order = np.argsort(-partners, kind="stable")[:N_FEATURES]
    if feature_frequency is None:
        feat_order = np.random.RandomState(seed).permutation(N_FEATURES)
    else:
        feat_order = np.argsort(-np.asarray(feature_frequency), kind="stable")
    f2i = np.zeros(N_FEATURES, dtype=np.int32)
    f2i[feat_order] = pn_order
    return f2i


class FlyEncoder:
    """Fixed fly-brain front end: PN -> KC with APL inhibition and KC recurrence."""

    def __init__(self, circuit: dict, feature_to_input: np.ndarray | None = None, theta: float = 1.0,
                 g_apl: float = 1.0, g_kk: float = 0.3, n_steps: int = 3, pn_total: float = 20.0, seed: int = 0):
        n_inputs = len(circuit["input_ids"])
        assert n_inputs >= N_FEATURES, "not enough KC-input neurons for one-per-feature mapping"
        if feature_to_input is None:
            feature_to_input = default_feature_map(circuit, None, seed)
        self.feature_to_input = np.asarray(feature_to_input, dtype=np.int32)
        self.pn_total = float(pn_total)
        # Each projection neuron injects one unit of drive into the calyx, split
        # across its Kenyon-cell synapses in proportion to synapse count; the
        # whole matrix is scaled so the mean Kenyon-cell drive is 1.
        W_in = _col_normalize(circuit["W_in"])[:, self.feature_to_input]
        self.W_in = (W_in * (W_in.shape[0] / self.pn_total)).tocsr().astype(np.float32)   # KC x features
        self.W_kk = _row_normalize(circuit["W_kk"])                # KC x KC
        self.W_ka = _row_normalize(circuit["W_ka"])                # APL x KC   (APL = weighted mean KC rate)
        W_ak = circuit["W_ak"].tocsr().astype(np.float32)          # KC x APL
        # Scale APL->KC so the average KC receives unit inhibition per unit APL rate.
        col_mean = np.asarray(W_ak.sum(axis=0)).ravel() / max(1, W_ak.shape[0])
        col_mean[col_mean == 0] = 1.0
        self.W_ak = W_ak.dot(sp.diags(1.0 / col_mean)).tocsr().astype(np.float32)
        self.theta, self.g_apl, self.g_kk, self.n_steps = float(theta), float(g_apl), float(g_kk), int(n_steps)
        self.n_kc = self.W_in.shape[0]

    def pn_rates(self, X: np.ndarray) -> np.ndarray:
        """Antennal-lobe style divisive normalisation (Olsen & Wilson 2010):
        total projection-neuron output is held constant whatever the number of
        active features, so sparse endgames drive the calyx as hard as
        crowded middlegames."""
        X = np.atleast_2d(X).astype(np.float32)
        n_active = X.sum(axis=1, keepdims=True)
        n_active[n_active == 0] = 1.0
        return X * (self.pn_total / n_active)

    def kc_from_features(self, X: np.ndarray) -> np.ndarray:
        """X: (B, 773) -> KC rates (B, n_kc)."""
        R = self.pn_rates(X)
        h = np.asarray(self.W_in.dot(R.T).T)                        # (B, KC) feed-forward drive
        kc = np.maximum(h - self.theta, 0.0)
        for _ in range(self.n_steps):
            apl = np.asarray(self.W_ka.dot(kc.T).T)                 # (B, 2)
            rec = np.asarray(self.W_kk.dot(kc.T).T)                 # (B, KC)
            inh = np.asarray(self.W_ak.dot(apl.T).T)                # (B, KC)
            kc = np.maximum(h + self.g_kk * rec - self.g_apl * inh - self.theta, 0.0)
        return kc.astype(np.float32)

    def encode(self, boards) -> np.ndarray:
        X = np.stack([board_features(b) for b in boards])
        return self.kc_from_features(X)

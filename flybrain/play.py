"""Play against the fly from the terminal, or measure its strength.

    python -m flybrain.play                  # you vs the fly (type moves in SAN)
    python -m flybrain.play --vs random -n 20 --depth 2
    python -m flybrain.play --vs stockfish --sf-skill 0 --sf-depth 1 -n 10
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import chess
import chess.engine
import numpy as np

from .connectome import load_circuit
from .encoder import FlyEncoder, board_features
from .model import MBONReadout

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


class FlyPlayer:
    """The fly brain plus a minimal lookahead over its own evaluations."""

    def __init__(self, circuit_path="data/circuit.npz", weights_path="data/fly_chess.npz", depth=2, jitter=0.0, seed=0):
        circuit = load_circuit(circuit_path)
        st = np.load(weights_path, allow_pickle=False)
        enc_kwargs = {k[4:]: float(st[k]) for k in st.files if k.startswith("enc_")}
        enc_kwargs["n_steps"] = int(enc_kwargs["n_steps"])
        self.encoder = FlyEncoder(circuit, feature_to_input=st["feature_to_input"], **enc_kwargs)
        self.readout = MBONReadout(circuit["W_km"], circuit["mbon_valence"], plasticity=str(st["plasticity"]))
        self.readout.load_state(st)
        self.depth, self.jitter, self.rng = depth, jitter, random.Random(seed)
        self.nodes = 0

    def evaluate(self, board: chess.Board) -> float:
        K = self.encoder.kc_from_features(board_features(board)[None, :])
        return float(self.readout.value(K)[0])

    def _terminal(self, board, depth):
        if board.is_checkmate():
            return -1.0 - 0.01 * depth
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        return None

    @staticmethod
    def _order(board):
        def key(m):
            s = 0
            if board.is_capture(m):
                victim = board.piece_type_at(m.to_square) or chess.PAWN
                s += 10 * PIECE_VALUE[victim] - PIECE_VALUE[board.piece_type_at(m.from_square)]
            if m.promotion:
                s += 8
            return -s
        return sorted(board.legal_moves, key=key)

    def _negamax(self, board, depth, alpha, beta):
        self.nodes += 1
        t = self._terminal(board, depth)
        if t is not None:
            return t
        if depth == 0:
            return self.evaluate(board)
        best = -np.inf
        for m in self._order(board):
            board.push(m)
            s = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            best = max(best, s)
            alpha = max(alpha, s)
            if alpha >= beta:
                break
        return best

    def evaluate_many(self, boards) -> np.ndarray:
        """Batched fly evaluation (same numbers as evaluate(), much faster)."""
        if not boards:
            return np.zeros(0, dtype=np.float32)
        K = self.encoder.encode(boards)
        return self.readout.value(K)

    def _choose_batched(self, board: chess.Board):
        """Depth-1/2 search with all leaf positions evaluated in one batch."""
        moves = self._order(board)
        leaves, meta = [], []     # meta: (move index, reply index or None)
        fixed = {}                # (move index) -> list of terminal leaf values
        for mi, m in enumerate(moves):
            board.push(m)
            t = self._terminal(board, self.depth - 1)
            if t is not None or self.depth == 1:
                if t is not None:
                    fixed[mi] = [-t]
                else:
                    leaves.append(board.copy(stack=False)); meta.append((mi, None))
            else:
                fixed.setdefault(mi, [])
                for r in board.legal_moves:
                    board.push(r)
                    t2 = self._terminal(board, self.depth - 2)
                    if t2 is not None:
                        fixed[mi].append(t2)              # our value at the leaf (we are to move)
                    else:
                        leaves.append(board.copy(stack=False)); meta.append((mi, len(fixed[mi])))
                    board.pop()
            board.pop()
        vals = self.evaluate_many(leaves)
        self.nodes += len(leaves)
        per_move = {mi: list(v) for mi, v in fixed.items()}
        for (mi, _), v in zip(meta, vals):
            per_move.setdefault(mi, []).append(float(v))
        scored = []
        for mi, m in enumerate(moves):
            vs = per_move.get(mi, [])
            if self.depth == 1:
                scored.append((vs[0] if vs else 0.0, m))
            else:
                scored.append((min(vs) if vs else 0.0, m))   # opponent picks our worst leaf
        return scored

    def choose(self, board: chess.Board, history=()):
        self.nodes = 0
        seen = set(history)
        scored = []
        if self.depth <= 2:
            scored = self._choose_batched(board)
            out = []
            for s, m in scored:
                board.push(m)
                if board.epd() in seen:
                    s -= 0.05
                board.pop()
                out.append((s, m))
            scored = out
        else:
            for m in self._order(board):
                board.push(m)
                s = -self._negamax(board, self.depth - 1, -np.inf, np.inf)
                if board.epd() in seen:
                    s -= 0.05
                board.pop()
                scored.append((s, m))
        scored.sort(key=lambda x: -x[0])
        top = scored[0][0]
        cands = [m for s, m in scored if s >= top - self.jitter]
        return self.rng.choice(cands), scored


def play_game(fly: FlyPlayer, opponent, fly_color: bool, max_plies=300):
    board = chess.Board()
    history = []
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        history.append(board.epd())
        if board.turn == fly_color:
            move, _ = fly.choose(board, history)
        else:
            move = opponent(board)
        board.push(move)
    res = board.result(claim_draw=True)
    if res == "1/2-1/2" or len(board.move_stack) >= max_plies:
        return 0.5
    winner_white = res == "1-0"
    return 1.0 if winner_white == (fly_color == chess.WHITE) else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit", default="data/circuit.npz")
    p.add_argument("--weights", default="data/fly_chess.npz")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--vs", choices=["human", "random", "stockfish"], default="human")
    p.add_argument("-n", "--games", type=int, default=10)
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH", "stockfish"))
    p.add_argument("--sf-skill", type=int, default=0)
    p.add_argument("--sf-depth", type=int, default=1)
    p.add_argument("--sf-elo", type=int, default=None, help="use UCI_LimitStrength with this Elo")
    args = p.parse_args()
    fly = FlyPlayer(args.circuit, args.weights, depth=args.depth, jitter=0.004)

    if args.vs == "human":
        board = chess.Board()
        human = chess.WHITE
        print("You are White. Enter moves in SAN (e.g. e4, Nf3, O-O). 'q' quits.")
        while not board.is_game_over(claim_draw=True):
            print(board, "\n")
            if board.turn == human:
                s = input("your move> ").strip()
                if s == "q":
                    return
                try:
                    board.push_san(s)
                except ValueError:
                    print("illegal move"); continue
            else:
                t = time.time()
                move, scored = fly.choose(board, [])
                print(f"fly plays {board.san(move)}  (value {scored[0][0]:+.3f}, {fly.nodes} positions imagined, {time.time() - t:.1f}s)")
                board.push(move)
        print(board, "\n", board.result(claim_draw=True))
        return

    rng = random.Random(1)
    engine = None
    if args.vs == "random":
        opponent = lambda b: rng.choice(list(b.legal_moves))
    else:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        opts = {"Skill Level": args.sf_skill}
        if args.sf_elo:
            opts = {"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo}
        engine.configure(opts)
        opponent = lambda b: engine.play(b, chess.engine.Limit(depth=args.sf_depth)).move
    scores = []
    t0 = time.time()
    for g in range(args.games):
        color = chess.WHITE if g % 2 == 0 else chess.BLACK
        s = play_game(fly, opponent, color)
        scores.append(s)
        print(f"game {g + 1}: fly as {'white' if color else 'black'} -> {s}  (running score {np.mean(scores):.2f}, {time.time() - t0:.0f}s)", flush=True)
    print(f"fly score vs {args.vs}: {np.mean(scores):.2f} over {len(scores)} games (win={scores.count(1.0)}, draw={scores.count(0.5)}, loss={scores.count(0.0)})")
    if engine:
        engine.quit()


if __name__ == "__main__":
    main()

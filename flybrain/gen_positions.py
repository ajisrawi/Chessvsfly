"""Generate a labelled chess-position dataset with Stockfish.

Each worker plays noisy self-play games (Stockfish's principal move most of
the time, a random legal move otherwise) and records every position it visits
together with Stockfish's evaluation *from the side to move's perspective*.

Output: CSV shards ``<out_dir>/positions-<worker>.csv`` with columns
``fen,cp`` where ``cp`` is centipawns (mate scores are clamped to +-3000).

The evaluations act as the "reward" signal (the dopamine surrogate) used to
train the fly's Kenyon-cell -> MBON synapses.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Process

import chess
import chess.engine

MATE_CP = 3000


def worker(idx: int, args: argparse.Namespace) -> None:
    rng = random.Random(args.seed * 1000 + idx)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": 1, "Hash": 16})
    path = os.path.join(args.out_dir, f"positions-{idx}.csv")
    n_written = 0
    t0 = time.time()
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if fh.tell() == 0:
            writer.writerow(["fen", "cp"])
        while n_written < args.per_worker:
            board = chess.Board()
            # A few random opening plies for diversity.
            for _ in range(rng.randint(0, 6)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            ply = 0
            while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
                info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                score = info["score"].relative
                cp = score.score(mate_score=MATE_CP)
                cp = max(-MATE_CP, min(MATE_CP, cp))
                writer.writerow([board.fen(), cp])
                n_written += 1
                pv = info.get("pv")
                if pv and rng.random() > args.random_move_prob:
                    move = pv[0]
                else:
                    move = rng.choice(list(board.legal_moves))
                board.push(move)
                ply += 1
                if n_written >= args.per_worker:
                    break
            fh.flush()
            if idx == 0 and n_written % 2000 < args.max_plies:
                rate = n_written / max(1e-9, time.time() - t0)
                print(f"[worker0] {n_written} positions, {rate:.0f}/s", file=sys.stderr, flush=True)
    engine.quit()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH", "stockfish"))
    p.add_argument("--out-dir", default="data/positions")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--per-worker", type=int, default=100_000)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--max-plies", type=int, default=140)
    p.add_argument("--random-move-prob", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    procs = [Process(target=worker, args=(i, args)) for i in range(args.workers)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()

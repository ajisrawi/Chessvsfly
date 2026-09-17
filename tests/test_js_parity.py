"""The browser engine must reproduce the Python fly brain."""
import json
import os
import shutil
import subprocess
import sys

import chess
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
WEB_DATA = os.path.join(ROOT, "web", "data", "fly_brain.json")
WEIGHTS = os.path.join(ROOT, "data", "fly_chess.npz")

FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
]


@pytest.mark.skipif(not (os.path.exists(WEB_DATA) and os.path.exists(WEIGHTS) and shutil.which("node")),
                    reason="needs exported brain and node")
def test_js_matches_python():
    from flybrain.play import FlyPlayer

    fly = FlyPlayer(os.path.join(ROOT, "data", "circuit.npz"), WEIGHTS)
    py = [fly.evaluate(chess.Board(f)) for f in FENS]
    out = subprocess.run(["node", os.path.join(ROOT, "web", "test", "parity.mjs"), json.dumps(FENS)],
                         capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    assert len(js) == len(py)
    for f, a, b in zip(FENS, py, js):
        assert abs(a - b["value"]) < 1e-3, (f, a, b)

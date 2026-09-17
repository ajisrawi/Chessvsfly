"""Train the fly's KC->MBON synapses to evaluate chess positions.

    python -m flybrain.train --positions data/positions --out data/fly_chess.npz

Steps: load Stockfish-labelled positions, run every position through the
fixed fly front end (PN -> KC), then fit the plastic KC->MBON synapses with
Adam on a squared error between the fly's output (tanh of MBON valence sum)
and tanh(cp / 500).
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from multiprocessing import Pool

import chess
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .connectome import load_circuit
from .encoder import FlyEncoder, board_features, default_feature_map
from .model import Adam, MBONReadout

CP_SCALE = 500.0
_ENC = None


def _init_worker(circuit_path, enc_kwargs, feature_to_input):
    global _ENC
    _ENC = FlyEncoder(load_circuit(circuit_path), feature_to_input=feature_to_input, **enc_kwargs)


def _encode_chunk(fens):
    X = np.stack([board_features(chess.Board(f)) for f in fens])
    return sp.csr_matrix(_ENC.kc_from_features(X))


def encode_all(fens, circuit_path, enc_kwargs, feature_to_input, workers, chunk=2000):
    chunks = [fens[i:i + chunk] for i in range(0, len(fens), chunk)]
    with Pool(workers, initializer=_init_worker, initargs=(circuit_path, enc_kwargs, feature_to_input)) as pool:
        mats = pool.map(_encode_chunk, chunks)
    return sp.vstack(mats).tocsr()


def load_positions(pattern, limit=None):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(pattern, "*.csv")))]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("fen")
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    if limit:
        df = df.iloc[:limit]
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--positions", default="data/positions")
    p.add_argument("--circuit", default="data/circuit.npz")
    p.add_argument("--out", default="data/fly_chess.npz")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cache", default=None, help="npz file to cache/restore the encoded positions")
    p.add_argument("--plasticity", choices=["existing", "all"], default="existing")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--val-frac", type=float, default=0.03)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--g-apl", type=float, default=1.0)
    p.add_argument("--g-kk", type=float, default=0.3)
    p.add_argument("--n-steps", type=int, default=3)
    p.add_argument("--pn-total", type=float, default=20.0)
    args = p.parse_args()

    enc_kwargs = dict(theta=args.theta, g_apl=args.g_apl, g_kk=args.g_kk, n_steps=args.n_steps, pn_total=args.pn_total)
    t0 = time.time()
    circuit = load_circuit(args.circuit)
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=False)
        K_all = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        y_all, feature_to_input = z["y"], z["feature_to_input"]
        print(f"restored {K_all.shape[0]} encoded positions from {args.cache}", flush=True)
    else:
        df = load_positions(args.positions, args.limit)
        y_all = np.tanh(df["cp"].to_numpy(dtype=np.float32) / CP_SCALE)
        # Which projection neuron "smells" which board feature: frequent features
        # get the best-connected neurons (estimated on a sample of positions).
        sample = df["fen"].iloc[:50000]
        freq = np.mean([board_features(chess.Board(f)) for f in sample], axis=0)
        feature_to_input = default_feature_map(circuit, freq)
        print(f"{len(df)} unique positions; encoding with {args.workers} workers ...", flush=True)
        K_all = encode_all(df["fen"].tolist(), args.circuit, enc_kwargs, feature_to_input, args.workers)
        print(f"encoded in {time.time() - t0:.0f}s; KC density {K_all.nnz / (K_all.shape[0] * K_all.shape[1]):.3f}", flush=True)
        if args.cache:
            np.savez(args.cache, data=K_all.data, indices=K_all.indices, indptr=K_all.indptr, shape=np.array(K_all.shape),
                     y=y_all, feature_to_input=feature_to_input)

    n_val = int(K_all.shape[0] * args.val_frac)
    K_val, y_val = K_all[:n_val], y_all[:n_val]
    K_tr, y_tr = K_all[n_val:], y_all[n_val:]

    model = MBONReadout(circuit["W_km"], circuit["mbon_valence"], plasticity=args.plasticity)
    model.calibrate(K_tr[:20000])
    opt = Adam([model.W, model.b], lr=args.lr)
    rng = np.random.RandomState(0)
    n = K_tr.shape[0]
    best = None
    for epoch in range(args.epochs):
        # learning rate decays geometrically to lr/20 by the last epoch
        opt.lr = args.lr * (0.05 ** (epoch / max(1, args.epochs - 1)))
        order = rng.permutation(n)
        tl, nb = 0.0, 0
        for i in range(0, n, args.batch):
            idx = order[i:i + args.batch]
            Kb, yb = K_tr[idx], y_tr[idx]
            v, cache = model.forward(Kb)
            tl += float(np.mean((v - yb) ** 2)); nb += 1
            dW, db = model.backward(cache, yb)
            opt.step([dW, db])
            np.maximum(model.W, 0.0, out=model.W)         # synapses stay excitatory
            model.W *= model.mask
        vv = model.value(K_val)
        mse = float(np.mean((vv - y_val) ** 2))
        r = float(np.corrcoef(vv, y_val)[0, 1])
        sign_acc = float(np.mean(np.sign(vv) == np.sign(y_val)))
        print(f"epoch {epoch + 1:3d}  train mse {tl / nb:.4f}  val mse {mse:.4f}  r {r:.3f}  sign acc {sign_acc:.3f}  "
              f"active synapses {int((model.W > 0).sum())}  [{time.time() - t0:.0f}s]", flush=True)
        if best is None or mse < best:
            best = mse
            np.savez_compressed(args.out, **model.state(), **{f"enc_{k}": np.array(v) for k, v in enc_kwargs.items()},
                                feature_to_input=feature_to_input, val_mse=np.array(mse), val_r=np.array(r),
                                n_train=np.array(n))
    print(f"saved {args.out} (best val mse {best:.4f})")


if __name__ == "__main__":
    main()

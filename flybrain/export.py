"""Export the trained fly brain for the browser.

Writes ``web/data/fly_brain.json`` (metadata + array offsets) and
``web/data/fly_brain.bin`` (little-endian typed arrays).  All synaptic
matrices are stored *by presynaptic neuron* so that the JavaScript engine can
propagate activity from active neurons only.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp

from .connectome import load_circuit
from .encoder import FlyEncoder, N_FEATURES


class BinWriter:
    def __init__(self):
        self.chunks, self.offset, self.arrays = [], 0, {}

    def add(self, name: str, arr: np.ndarray):
        arr = np.ascontiguousarray(arr)
        dtype = {np.dtype("float32"): "f32", np.dtype("int32"): "i32", np.dtype("uint16"): "u16",
                 np.dtype("uint8"): "u8"}[arr.dtype]
        pad = (-self.offset) % 4
        if pad:
            self.chunks.append(b"\0" * pad); self.offset += pad
        self.arrays[name] = {"dtype": dtype, "offset": self.offset, "length": int(arr.size)}
        b = arr.tobytes(); self.chunks.append(b); self.offset += len(b)

    def bytes(self):
        return b"".join(self.chunks)


def by_source(m: sp.csr_matrix, idx_dtype):
    """Return (indptr, indices, data) of m indexed by column (presynaptic neuron)."""
    c = m.tocsc()
    return c.indptr.astype(np.int32), c.indices.astype(idx_dtype), c.data.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit", default="data/circuit.npz")
    p.add_argument("--weights", default="data/fly_chess.npz")
    p.add_argument("--out-dir", default="web/data")
    args = p.parse_args()

    circuit = load_circuit(args.circuit)
    st = np.load(args.weights, allow_pickle=False)
    enc_kwargs = {k[4:]: float(st[k]) for k in st.files if k.startswith("enc_")}
    enc_kwargs["n_steps"] = int(enc_kwargs["n_steps"])
    enc = FlyEncoder(circuit, feature_to_input=st["feature_to_input"], **enc_kwargs)
    W = st["W"].astype(np.float32) * st["mask"].astype(np.float32)

    w = BinWriter()
    ip, ix, d = by_source(enc.W_in, np.uint16); w.add("in_indptr", ip); w.add("in_idx", ix); w.add("in_val", d)
    ip, ix, d = by_source(enc.W_kk, np.uint16); w.add("kk_indptr", ip); w.add("kk_idx", ix); w.add("kk_val", d)
    w.add("ka", enc.W_ka.toarray().astype(np.float32).ravel())              # (2, KC)
    w.add("ak", enc.W_ak.toarray().astype(np.float32).ravel())              # (KC, 2)
    ip, ix, d = by_source(sp.csr_matrix(W), np.uint8); w.add("km_indptr", ip); w.add("km_idx", ix); w.add("km_val", d)
    w.add("mbon_bias", st["b"].astype(np.float32))
    w.add("mbon_valence", st["valence"].astype(np.float32))

    meta = {
        "dataset": "FlyEM Male CNS connectome v1.0 (gs://flyem-male-cns, CC-BY)",
        "n_features": N_FEATURES,
        "n_kc": int(enc.n_kc),
        "n_mbon": int(W.shape[0]),
        "n_apl": int(enc.W_ka.shape[0]),
        "n_inputs_available": int(len(circuit["input_ids"])),
        "params": {"theta": enc.theta, "g_apl": enc.g_apl, "g_kk": enc.g_kk, "n_steps": enc.n_steps,
                   "pn_total": enc.pn_total},
        "feature_to_input": enc.feature_to_input.tolist(),
        "input_body_ids": circuit["input_ids"].tolist(),
        "input_types": circuit["input_types"].tolist(),
        "kc_body_ids": circuit["kc_ids"].tolist(),
        "kc_types": circuit["kc_types"].tolist(),
        "mbon_body_ids": circuit["mbon_ids"].tolist(),
        "mbon_types": circuit["mbon_types"].tolist(),
        "mbon_nt": circuit["mbon_nt"].tolist(),
        "apl_body_ids": circuit["apl_ids"].tolist(),
        "synapse_counts": {
            "pn_to_kc": int(circuit["W_in"].sum()), "kc_to_kc": int(circuit["W_kk"].sum()),
            "kc_to_apl": int(circuit["W_ka"].sum()), "apl_to_kc": int(circuit["W_ak"].sum()),
            "kc_to_mbon": int(circuit["W_km"].sum()),
        },
        "trained_synapses": int((W > 0).sum()),
        "val_mse": float(st["val_mse"]) if "val_mse" in st.files else None,
        "val_r": float(st["val_r"]) if "val_r" in st.files else None,
        "arrays": w.arrays,
        "bin": "fly_brain.bin",
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "fly_brain.bin"), "wb") as fh:
        fh.write(w.bytes())
    with open(os.path.join(args.out_dir, "fly_brain.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"wrote {args.out_dir}/fly_brain.bin ({w.offset / 1e6:.1f} MB) and fly_brain.json")


if __name__ == "__main__":
    main()

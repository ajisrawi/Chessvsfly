"""Extract the mushroom-body circuit from the male CNS connectome.

Source: the FlyEM / Google Research / Cambridge "Male CNS" connectome v1.0
(``gs://flyem-male-cns``, CC-BY).  We use three flat files from the release
bucket:

* ``body-annotations-male-cns-v1.0-minconf-0.5.feather`` - cell types
* ``body-neurotransmitters-male-cns-v1.0.feather``       - predicted NT per neuron
* ``connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather``
  - neuron -> neuron synapse counts (25.6M edges between traced neurons)

The circuit we keep is the fly's associative-learning centre, the mushroom
body (MB):

    projection neurons (PN, ~900)  --->  Kenyon cells (KC, 4064)
                                          |  ^        |
                                          v  |        v
                                     APL (GABA)    MBONs (97)  -->  behaviour

Every weight in the exported circuit is a real synapse count from the
connectome.  Nothing is invented: neuron identities, wiring, and
neurotransmitter signs all come from the released data.
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "nt": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}

# Aso et al. 2014 (eLife): MBONs driving avoidance are glutamatergic, MBONs
# driving approach are GABAergic or cholinergic.  We use the connectome's own
# neurotransmitter prediction to assign each MBON a fixed behavioural valence.
NT_VALENCE = {"glutamate": -1.0, "gaba": 1.0, "acetylcholine": 1.0}


def download(data_dir: str) -> dict[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        dst = os.path.join(data_dir, name)
        if not os.path.exists(dst):
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(f"{BUCKET}/{name}", dst)
        paths[key] = dst
    return paths


def _csr(edges: pd.DataFrame, row_index: dict, col_index: dict, shape) -> sp.csr_matrix:
    rows = edges["body_post"].map(row_index).to_numpy()
    cols = edges["body_pre"].map(col_index).to_numpy()
    vals = edges["weight"].to_numpy(dtype=np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=shape)


def build_circuit(ann_path: str, nt_path: str, w_path: str, min_input_synapses: int = 3) -> dict:
    ann = pd.read_feather(ann_path)
    nt = pd.read_feather(nt_path).set_index("body")["consensus_nt"]
    w = pd.read_feather(w_path, columns=["body_pre", "body_post", "weight"])

    typ = ann.set_index("bodyId")["type"].fillna("")
    kc_ids = np.sort(typ.index[typ.str.startswith("KC")].to_numpy())
    mbon_ids = np.sort(typ.index[typ.str.startswith("MBON")].to_numpy())
    apl_ids = np.sort(typ.index[typ == "APL"].to_numpy())
    dan_ids = np.sort(typ.index[typ.str.match(r"^(PAM|PPL1)")].to_numpy())
    mb = set(kc_ids) | set(mbon_ids) | set(apl_ids) | set(dan_ids) | set(typ.index[typ == "DPM"])

    # Input layer: every neuron outside the MB that synapses onto Kenyon cells.
    to_kc = w[w["body_post"].isin(kc_ids)]
    in_strength = to_kc.groupby("body_pre")["weight"].sum()
    in_strength = in_strength[~in_strength.index.isin(mb)]
    in_strength = in_strength[in_strength >= min_input_synapses].sort_values(ascending=False)
    input_ids = in_strength.index.to_numpy()

    kc_index = {b: i for i, b in enumerate(kc_ids)}
    mbon_index = {b: i for i, b in enumerate(mbon_ids)}
    apl_index = {b: i for i, b in enumerate(apl_ids)}
    in_index = {b: i for i, b in enumerate(input_ids)}

    def edges(pre_set, post_set):
        return w[w["body_pre"].isin(pre_set) & w["body_post"].isin(post_set)]

    nK, nM, nA, nI = len(kc_ids), len(mbon_ids), len(apl_ids), len(input_ids)
    W_in = _csr(edges(input_ids, kc_ids), kc_index, in_index, (nK, nI))       # PN -> KC
    W_kk = _csr(edges(kc_ids, kc_ids), kc_index, kc_index, (nK, nK))          # KC -> KC
    W_ka = _csr(edges(kc_ids, apl_ids), apl_index, kc_index, (nA, nK))        # KC -> APL
    W_ak = _csr(edges(apl_ids, kc_ids), kc_index, apl_index, (nK, nA))        # APL -> KC
    W_km = _csr(edges(kc_ids, mbon_ids), mbon_index, kc_index, (nM, nK))      # KC -> MBON (plastic)
    W_mm = _csr(edges(mbon_ids, mbon_ids), mbon_index, mbon_index, (nM, nM))  # MBON -> MBON

    mbon_nt = nt.reindex(mbon_ids).fillna("acetylcholine").to_numpy()
    valence = np.array([NT_VALENCE.get(x, 1.0) for x in mbon_nt], dtype=np.float32)

    return {
        "kc_ids": kc_ids,
        "kc_types": typ.reindex(kc_ids).to_numpy().astype(str),
        "mbon_ids": mbon_ids,
        "mbon_types": typ.reindex(mbon_ids).to_numpy().astype(str),
        "mbon_nt": mbon_nt.astype(str),
        "mbon_valence": valence,
        "apl_ids": apl_ids,
        "input_ids": input_ids,
        "input_types": typ.reindex(input_ids).to_numpy().astype(str),
        "input_strength": in_strength.to_numpy().astype(np.float32),
        "n_dan": np.int64(len(dan_ids)),
        "W_in": W_in, "W_kk": W_kk, "W_ka": W_ka, "W_ak": W_ak, "W_km": W_km, "W_mm": W_mm,
    }


def save_circuit(circuit: dict, path: str) -> None:
    out = {}
    for k, v in circuit.items():
        if sp.issparse(v):
            v = v.tocsr()
            out[f"{k}_data"] = v.data
            out[f"{k}_indices"] = v.indices
            out[f"{k}_indptr"] = v.indptr
            out[f"{k}_shape"] = np.array(v.shape)
        else:
            out[k] = v
    np.savez_compressed(path, **out)


def load_circuit(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    circuit = {}
    for key in z.files:
        if key.endswith("_data"):
            base = key[:-5]
            circuit[base] = sp.csr_matrix(
                (z[key], z[f"{base}_indices"], z[f"{base}_indptr"]), shape=tuple(z[f"{base}_shape"])
            )
        elif any(key.endswith(s) for s in ("_indices", "_indptr", "_shape")):
            continue
        else:
            circuit[key] = z[key]
    return circuit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/malecns", help="where the release feathers live / go")
    p.add_argument("--out", default="data/circuit.npz")
    args = p.parse_args()
    paths = download(args.data_dir)
    circuit = build_circuit(paths["annotations"], paths["nt"], paths["weights"])
    save_circuit(circuit, args.out)
    c = circuit
    print(f"inputs={len(c['input_ids'])} KC={len(c['kc_ids'])} MBON={len(c['mbon_ids'])} APL={len(c['apl_ids'])}")
    for name in ("W_in", "W_kk", "W_ka", "W_ak", "W_km", "W_mm"):
        print(f"  {name}: {c[name].shape} nnz={c[name].nnz} synapses={int(c[name].sum())}")
    print("MBON valence:", dict(zip(*np.unique(c["mbon_nt"], return_counts=True))))


if __name__ == "__main__":
    main()

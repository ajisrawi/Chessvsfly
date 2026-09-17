"""The plastic part of the fly brain: Kenyon cell -> MBON synapses.

Biology (Aso et al. 2014, Hige et al. 2015): the mushroom body learns by
dopamine-gated plasticity of the *excitatory* (cholinergic) KC->MBON synapses.
MBONs then bias behaviour: glutamatergic MBONs drive avoidance, cholinergic and
GABAergic MBONs drive approach.

Model:
    mbon_j = relu( sum_i W_ji * kc_i + b_j )      W_ji >= 0, only where the
                                                   connectome has a synapse
    value  = tanh( sum_j valence_j * mbon_j )      valence fixed by
                                                   neurotransmitter (+-1)

We train W and b with a supervised surrogate for dopamine: Stockfish's
evaluation of the position is the reward signal.  Weights are clipped at zero
after every update (synapses can weaken to silence but never change sign).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class MBONReadout:
    def __init__(self, W_km: sp.csr_matrix, valence: np.ndarray, plasticity: str = "existing",
                 init_scale: float = 0.05, seed: int = 0):
        self.n_mbon, self.n_kc = W_km.shape
        self.valence = valence.astype(np.float32)
        W0 = W_km.toarray().astype(np.float32)
        rows = W0.sum(axis=1, keepdims=True)
        rows[rows == 0] = 1.0
        W0 = W0 / rows                                  # each MBON: connectome-weighted mean of its KCs
        self.mask = (W0 > 0).astype(np.float32) if plasticity == "existing" else np.ones_like(W0)
        rng = np.random.RandomState(seed)
        if plasticity == "existing":
            self.W = (W0 * init_scale * self.n_kc / np.maximum(1, self.mask.sum(1, keepdims=True))).astype(np.float32)
        else:
            self.W = (rng.rand(*W0.shape) * init_scale / 20).astype(np.float32)
        self.b = np.zeros(self.n_mbon, dtype=np.float32)
        self.plasticity = plasticity

    def calibrate(self, K, target_std: float = 0.5):
        """Scale the initial synapses so the pre-tanh output has a sensible spread.

        Keeps the relative connectome-derived strengths but puts the fly in the
        responsive part of its output range before learning starts."""
        _, (_, _, m, _) = self.forward(K)
        s = m @ self.valence
        std = float(np.std(s))
        if std > 0:
            self.W *= target_std / std
        return self

    # ---- forward -------------------------------------------------------
    def forward(self, K):
        """K: sparse or dense (B, n_kc) -> (value (B,), cache)."""
        z = np.asarray(K.dot(self.W.T)) + self.b            # (B, M)
        m = np.maximum(z, 0.0)
        s = m @ self.valence
        v = np.tanh(s)
        return v, (K, z, m, v)

    def value(self, K):
        return self.forward(K)[0]

    def mbon_rates(self, K):
        return self.forward(K)[1][2]

    # ---- backward ------------------------------------------------------
    def backward(self, cache, y):
        K, z, m, v = cache
        B = y.shape[0]
        dv = (2.0 / B) * (v - y)
        ds = dv * (1.0 - v * v)
        dm = ds[:, None] * self.valence[None, :]
        dz = dm * (z > 0)
        dW = np.asarray(K.T.dot(dz)).T if sp.issparse(K) else dz.T @ K
        db = dz.sum(axis=0)
        return dW.astype(np.float32) * self.mask, db.astype(np.float32)

    # ---- persistence ---------------------------------------------------
    def state(self):
        return {"W": self.W, "b": self.b, "valence": self.valence, "mask": self.mask,
                "plasticity": np.array(self.plasticity)}

    def load_state(self, st):
        self.W = st["W"].astype(np.float32)
        self.b = st["b"].astype(np.float32)
        self.valence = st["valence"].astype(np.float32)
        self.mask = st["mask"].astype(np.float32)


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.lr, self.b1, self.b2, self.eps, self.t = lr, beta1, beta2, eps, 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * (g * g)
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

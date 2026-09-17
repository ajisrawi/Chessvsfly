/**
 * The fly brain, in the browser.
 *
 * A JavaScript mirror of flybrain/encoder.py + flybrain/model.py.  Every
 * number it multiplies by is a synapse count from the male CNS connectome
 * (or one of the KC->MBON synapses after training).
 *
 * Usage:
 *   const brain = await FlyBrain.load('data/');   // fetches fly_brain.json/.bin
 *   const out = brain.evaluate(chess);            // chess: a chess.js instance
 *   out.value   -> [-1, 1], from the side to move's point of view
 *   out.kc      -> Float32Array of Kenyon-cell rates
 *   out.mbon    -> Float32Array of MBON rates
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FlyBrain = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const PIECE_INDEX = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
  const N_SQUARE_FEATURES = 64 * 12;
  const FILES = 'abcdefgh';

  class FlyBrain {
    constructor(meta, buffer) {
      this.meta = meta;
      const view = (name) => {
        const a = meta.arrays[name];
        const C = { f32: Float32Array, i32: Int32Array, u16: Uint16Array, u8: Uint8Array }[a.dtype];
        return new C(buffer, a.offset, a.length);
      };
      this.inIndptr = view('in_indptr'); this.inIdx = view('in_idx'); this.inVal = view('in_val');
      this.kkIndptr = view('kk_indptr'); this.kkIdx = view('kk_idx'); this.kkVal = view('kk_val');
      this.ka = view('ka'); this.ak = view('ak');
      this.kmIndptr = view('km_indptr'); this.kmIdx = view('km_idx'); this.kmVal = view('km_val');
      this.bias = view('mbon_bias'); this.valence = view('mbon_valence');
      this.nKC = meta.n_kc; this.nMBON = meta.n_mbon; this.nAPL = meta.n_apl;
      const p = meta.params;
      this.theta = p.theta; this.gApl = p.g_apl; this.gKK = p.g_kk; this.nSteps = p.n_steps; this.pnTotal = p.pn_total;
      // scratch buffers
      this.h = new Float32Array(this.nKC);
      this.kc = new Float32Array(this.nKC);
      this.kc2 = new Float32Array(this.nKC);
      this.rec = new Float32Array(this.nKC);
      this.apl = new Float32Array(this.nAPL);
      this.mbon = new Float32Array(this.nMBON);
      this.active = new Int32Array(this.nKC);
    }

    static async load(baseUrl, fetchImpl) {
      const f = fetchImpl || fetch;
      const meta = await (await f(baseUrl + 'fly_brain.json')).json();
      const buffer = await (await f(baseUrl + meta.bin)).arrayBuffer();
      return new FlyBrain(meta, buffer);
    }

    /** Active board features from the side to move's perspective (see encoder.py). */
    static features(chess) {
      const turn = chess.turn();
      const flip = turn === 'b';
      const board = chess.board();          // board()[0] is rank 8
      const feats = [];
      for (let r = 0; r < 8; r++) {
        for (let f = 0; f < 8; f++) {
          const piece = board[r][f];
          if (!piece) continue;
          const rank = 7 - r;                 // 0 = rank 1
          const sq = (flip ? 7 - rank : rank) * 8 + f;
          const ch = PIECE_INDEX[piece.type] + (piece.color === turn ? 0 : 6);
          feats.push(sq * 12 + ch);
        }
      }
      const fen = chess.fen().split(' ');
      const castling = fen[2] || '-';
      const us = turn === 'w' ? ['K', 'Q'] : ['k', 'q'];
      const them = turn === 'w' ? ['k', 'q'] : ['K', 'Q'];
      if (castling.includes(us[0])) feats.push(N_SQUARE_FEATURES + 0);
      if (castling.includes(us[1])) feats.push(N_SQUARE_FEATURES + 1);
      if (castling.includes(them[0])) feats.push(N_SQUARE_FEATURES + 2);
      if (castling.includes(them[1])) feats.push(N_SQUARE_FEATURES + 3);
      if (fen[3] && fen[3] !== '-') feats.push(N_SQUARE_FEATURES + 4);
      return feats;
    }

    /** Run the circuit on a list of active feature indices. Returns value in [-1, 1]. */
    evaluateFeatures(feats) {
      const { h, kc, kc2, rec, apl, mbon, active } = this;
      const nKC = this.nKC, nAPL = this.nAPL;
      h.fill(0);
      const rate = this.pnTotal / Math.max(1, feats.length);   // antennal-lobe gain control
      for (const f of feats) {
        for (let p = this.inIndptr[f]; p < this.inIndptr[f + 1]; p++) h[this.inIdx[p]] += this.inVal[p] * rate;
      }
      let nActive = 0;
      for (let k = 0; k < nKC; k++) {
        const v = h[k] - this.theta;
        kc[k] = v > 0 ? v : 0;
        if (v > 0) active[nActive++] = k;
      }
      for (let step = 0; step < this.nSteps; step++) {
        // APL neurons integrate the whole KC population
        for (let a = 0; a < nAPL; a++) {
          let s = 0; const off = a * nKC;
          for (let i = 0; i < nActive; i++) { const k = active[i]; s += this.ka[off + k] * kc[k]; }
          apl[a] = s;
        }
        // KC -> KC recurrence, propagated from active cells only
        rec.fill(0);
        for (let i = 0; i < nActive; i++) {
          const k = active[i], x = kc[k];
          for (let p = this.kkIndptr[k]; p < this.kkIndptr[k + 1]; p++) rec[this.kkIdx[p]] += this.kkVal[p] * x;
        }
        nActive = 0;
        for (let k = 0; k < nKC; k++) {
          let inh = 0;
          for (let a = 0; a < nAPL; a++) inh += this.ak[k * nAPL + a] * apl[a];
          const v = h[k] + this.gKK * rec[k] - this.gApl * inh - this.theta;
          kc2[k] = v > 0 ? v : 0;
          if (v > 0) active[nActive++] = k;
        }
        kc.set(kc2);
      }
      // KC -> MBON (the trained synapses)
      mbon.set(this.bias);
      for (let i = 0; i < nActive; i++) {
        const k = active[i], x = kc[k];
        for (let p = this.kmIndptr[k]; p < this.kmIndptr[k + 1]; p++) mbon[this.kmIdx[p]] += this.kmVal[p] * x;
      }
      let s = 0;
      for (let m = 0; m < this.nMBON; m++) { if (mbon[m] < 0) mbon[m] = 0; s += this.valence[m] * mbon[m]; }
      return { value: Math.tanh(s), nActive, drive: s };
    }

    evaluate(chess) {
      const out = this.evaluateFeatures(FlyBrain.features(chess));
      return { value: out.value, drive: out.drive, nActive: out.nActive, kc: this.kc, mbon: this.mbon, apl: this.apl };
    }
  }

  FlyBrain.PIECE_INDEX = PIECE_INDEX;
  return FlyBrain;
});

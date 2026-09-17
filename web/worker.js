// Web Worker: hosts the fly brain and searches moves with it.
// The only position evaluator is FlyBrain (the male CNS mushroom body);
// the worker just enumerates legal moves and lets the fly "imagine" each one.
import { Chess } from './vendor/chess.js';
import './flybrain.js';

let brain = null;
let nodes = 0;
const PIECE_VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

function orderMoves(moves) {
  // captures and promotions first: cheap ordering that helps alpha-beta prune
  return moves
    .map((m) => ({ m, s: (m.captured ? 10 * PIECE_VALUE[m.captured] - PIECE_VALUE[m.piece] : 0) + (m.promotion ? 8 : 0) }))
    .sort((a, b) => b.s - a.s)
    .map((x) => x.m);
}

function terminal(chess, depth) {
  if (chess.isCheckmate()) return -1 - 0.01 * depth; // being mated; sooner is worse
  if (chess.isDraw() || chess.isStalemate() || chess.isInsufficientMaterial()) return 0;
  return null;
}

function negamax(chess, depth, alpha, beta) {
  nodes++;
  const t = terminal(chess, depth);
  if (t !== null) return t;
  if (depth === 0) return brain.evaluateFeatures(FlyBrain.features(chess)).value;
  let best = -Infinity;
  for (const m of orderMoves(chess.moves({ verbose: true }))) {
    chess.move(m);
    const s = -negamax(chess, depth - 1, -beta, -alpha);
    chess.undo();
    if (s > best) best = s;
    if (s > alpha) alpha = s;
    if (alpha >= beta) break;
  }
  return best;
}

function chooseMove(fen, depth, history, jitter) {
  const chess = new Chess(fen);
  nodes = 0;
  const t0 = performance.now();
  const seen = new Set(history || []);
  const scored = [];
  for (const m of orderMoves(chess.moves({ verbose: true }))) {
    chess.move(m);
    let s = -negamax(chess, depth - 1, -Infinity, Infinity);
    const key = chess.fen().split(' ').slice(0, 4).join(' ');
    if (seen.has(key)) s -= 0.05; // a fly gets bored of repeating itself
    chess.undo();
    scored.push({ san: m.san, from: m.from, to: m.to, promotion: m.promotion, value: s });
  }
  scored.sort((a, b) => b.value - a.value);
  const top = scored[0].value;
  const candidates = scored.filter((x) => x.value >= top - (jitter || 0));
  const pick = candidates[Math.floor(Math.random() * candidates.length)];
  return { move: pick, scored, nodes, ms: performance.now() - t0 };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      brain = await FlyBrain.load(msg.baseUrl);
      const m = brain.meta;
      self.postMessage({
        type: 'ready',
        meta: {
          n_kc: m.n_kc, n_mbon: m.n_mbon, n_apl: m.n_apl, n_features: m.n_features,
          kc_types: m.kc_types, mbon_types: m.mbon_types, mbon_nt: m.mbon_nt,
          synapse_counts: m.synapse_counts, trained_synapses: m.trained_synapses,
          val_r: m.val_r, dataset: m.dataset, params: m.params,
        },
      });
    } else if (msg.type === 'eval') {
      const chess = new Chess(msg.fen);
      const out = brain.evaluate(chess);
      self.postMessage({ type: 'eval', id: msg.id, fen: msg.fen, value: out.value, nActive: out.nActive,
        kc: Float32Array.from(out.kc), mbon: Float32Array.from(out.mbon), apl: Float32Array.from(out.apl) });
    } else if (msg.type === 'move') {
      const res = chooseMove(msg.fen, msg.depth, msg.history, msg.jitter);
      self.postMessage({ type: 'move', id: msg.id, ...res });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.stack ? err.stack : err) });
  }
};

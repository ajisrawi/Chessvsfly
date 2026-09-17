import { Chess } from './vendor/chess.js';

const $ = (id) => document.getElementById(id);
const GLYPH = { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚' };
const FILES = 'abcdefgh';

const state = {
  chess: new Chess(),
  human: 'w',
  depth: 2,
  selected: null,
  thinking: false,
  ready: false,
  meta: null,
  lastMove: null,
  reqId: 0,
};

const worker = new Worker('worker.js', { type: 'module' });
const pending = new Map();
worker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'ready') { state.ready = true; state.meta = msg.meta; onReady(); return; }
  if (msg.type === 'error') { setStatus('The fly brain crashed: ' + msg.message); console.error(msg.message); return; }
  const cb = pending.get(msg.id);
  if (cb) { pending.delete(msg.id); cb(msg); }
};
function ask(msg) {
  return new Promise((resolve) => { const id = ++state.reqId; pending.set(id, resolve); worker.postMessage({ ...msg, id }); });
}
worker.postMessage({ type: 'init', baseUrl: new URL('data/', location.href).href });

// ---------------------------------------------------------------- board UI
function setStatus(text) { $('status').textContent = text; }

function renderBoard() {
  const board = $('board');
  board.innerHTML = '';
  const flip = state.human === 'b';
  const legalTargets = new Set();
  if (state.selected) for (const m of state.chess.moves({ square: state.selected, verbose: true })) legalTargets.add(m.to);
  const grid = state.chess.board();
  const inCheck = state.chess.inCheck();
  for (let rr = 0; rr < 8; rr++) {
    for (let ff = 0; ff < 8; ff++) {
      const r = flip ? 7 - rr : rr, f = flip ? 7 - ff : ff;
      const sq = FILES[f] + (8 - r);
      const piece = grid[r][f];
      const cell = document.createElement('div');
      cell.className = 'sq ' + ((r + f) % 2 === 0 ? 'light' : 'dark');
      cell.dataset.sq = sq;
      if (state.selected === sq) cell.classList.add('selected');
      if (legalTargets.has(sq)) cell.classList.add(piece ? 'capture' : 'target');
      if (state.lastMove && (state.lastMove.from === sq || state.lastMove.to === sq)) cell.classList.add('last');
      if (piece) {
        const p = document.createElement('span');
        p.className = 'piece ' + (piece.color === 'w' ? 'white' : 'black');
        p.textContent = GLYPH[piece.type];
        if (inCheck && piece.type === 'k' && piece.color === state.chess.turn()) cell.classList.add('check');
        cell.appendChild(p);
      }
      if (ff === 0) { const l = document.createElement('i'); l.className = 'coord rank'; l.textContent = 8 - r; cell.appendChild(l); }
      if (rr === 7) { const l = document.createElement('i'); l.className = 'coord file'; l.textContent = FILES[f]; cell.appendChild(l); }
      cell.addEventListener('click', () => onSquareClick(sq));
      board.appendChild(cell);
    }
  }
}

function renderMoves() {
  const hist = state.chess.history();
  const out = [];
  for (let i = 0; i < hist.length; i += 2) out.push(`<span class="n">${i / 2 + 1}.</span> ${hist[i]} ${hist[i + 1] || ''}`);
  $('moves').innerHTML = out.join(' ');
  $('moves').scrollTop = $('moves').scrollHeight;
}

function onSquareClick(sq) {
  if (!state.ready || state.thinking || state.chess.isGameOver()) return;
  if (state.chess.turn() !== state.human) return;
  const piece = state.chess.get(sq);
  if (state.selected) {
    const legal = state.chess.moves({ square: state.selected, verbose: true }).find((m) => m.to === sq);
    if (legal) {
      const mv = state.chess.move({ from: state.selected, to: sq, promotion: 'q' });
      state.selected = null;
      state.lastMove = mv;
      afterMove();
      return;
    }
  }
  state.selected = piece && piece.color === state.human ? sq : null;
  renderBoard();
}

function gameOverText() {
  const c = state.chess;
  if (c.isCheckmate()) return c.turn() === state.human ? 'Checkmate. The fly wins! 🪰🏆' : 'Checkmate. You beat a fly brain. 🏆';
  if (c.isStalemate()) return 'Stalemate. Draw.';
  if (c.isThreefoldRepetition()) return 'Draw by repetition.';
  if (c.isInsufficientMaterial()) return 'Draw: insufficient material.';
  if (c.isDraw()) return 'Draw.';
  return null;
}

async function afterMove() {
  renderBoard(); renderMoves();
  updateBrainView();
  const over = gameOverText();
  if (over) { setStatus(over); return; }
  if (state.chess.turn() !== state.human) await flyMove();
  else setStatus(state.chess.inCheck() ? 'Your move. You are in check!' : 'Your move.');
}

function positionKeys() {
  // all positions in the game so far (for the fly's repetition aversion)
  const c = new Chess();
  const keys = [c.fen().split(' ').slice(0, 4).join(' ')];
  for (const m of state.chess.history()) { c.move(m); keys.push(c.fen().split(' ').slice(0, 4).join(' ')); }
  return keys;
}

async function flyMove() {
  state.thinking = true;
  setStatus(`The fly is thinking (${state.depth} ply)…`);
  document.body.classList.add('thinking');
  const res = await ask({ type: 'move', fen: state.chess.fen(), depth: state.depth, history: positionKeys(), jitter: 0.004 });
  document.body.classList.remove('thinking');
  state.thinking = false;
  if (!res || !res.move) { setStatus('The fly could not find a move.'); return; }
  const mv = state.chess.move({ from: res.move.from, to: res.move.to, promotion: res.move.promotion });
  state.lastMove = mv;
  const top = res.scored.slice(0, 3).map((m) => `${m.san} ${(m.value >= 0 ? '+' : '') + m.value.toFixed(2)}`).join(', ');
  $('think-stats').textContent = `Imagined ${res.nodes.toLocaleString()} positions in ${(res.ms / 1000).toFixed(1)} s. Favourites: ${top}.`;
  renderBoard(); renderMoves();
  updateBrainView();
  const over = gameOverText();
  if (over) { setStatus(over); return; }
  setStatus(`The fly played ${mv.san}. ${state.chess.inCheck() ? 'Check! ' : ''}Your move.`);
}

// ---------------------------------------------------------------- brain view
const KC_COLORS = { g: '#7bd88f', ab: '#79b8ff', apbp: '#ffb86b', other: '#c8c8c8' };
function kcClass(t) {
  if (t.startsWith('KCg')) return 'g';
  if (t.startsWith("KCa'b'")) return 'apbp';
  if (t.startsWith('KCab')) return 'ab';
  return 'other';
}

function drawLegend() {
  $('kc-legend').innerHTML = [
    ['g', 'γ Kenyon cells'], ['ab', 'α/β Kenyon cells'], ['apbp', "α'/β' Kenyon cells"],
  ].map(([k, l]) => `<span><i class="sw" style="background:${KC_COLORS[k]}"></i>${l}</span>`).join('');
}

async function updateBrainView() {
  if (!state.ready) return;
  const res = await ask({ type: 'eval', fen: state.chess.fen() });
  if (res.fen !== state.chess.fen()) return; // stale
  const kcCanvas = $('kc'); const ctx = kcCanvas.getContext('2d');
  const n = res.kc.length, cols = 64, rows = Math.ceil(n / cols), cw = kcCanvas.width / cols, ch = kcCanvas.height / rows;
  ctx.fillStyle = '#0d0f14'; ctx.fillRect(0, 0, kcCanvas.width, kcCanvas.height);
  let max = 0; for (let i = 0; i < n; i++) if (res.kc[i] > max) max = res.kc[i];
  for (let i = 0; i < n; i++) {
    const v = res.kc[i];
    const x = (i % cols) * cw, y = Math.floor(i / cols) * ch;
    ctx.globalAlpha = v > 0 ? 0.35 + 0.65 * Math.min(1, v / (max || 1)) : 0.08;
    ctx.fillStyle = KC_COLORS[kcClass(state.meta.kc_types[i])];
    ctx.fillRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
  }
  ctx.globalAlpha = 1;
  $('kc-count').textContent = `${res.nActive} of ${n} firing`;

  const mc = $('mbon'); const mctx = mc.getContext('2d');
  mctx.fillStyle = '#0d0f14'; mctx.fillRect(0, 0, mc.width, mc.height);
  const m = res.mbon.length, bw = mc.width / m;
  let mmax = 0; for (let i = 0; i < m; i++) if (res.mbon[i] > mmax) mmax = res.mbon[i];
  for (let i = 0; i < m; i++) {
    const h = mmax > 0 ? (res.mbon[i] / mmax) * (mc.height - 4) : 0;
    mctx.fillStyle = state.meta.mbon_nt[i] === 'glutamate' ? '#ff6b6b' : '#7bd88f';
    mctx.fillRect(i * bw + 0.5, mc.height - h, bw - 1, h);
  }
  const v = res.value; // from the side to move's perspective
  const sideToMoveIsHuman = state.chess.turn() === state.human;
  $('mood-fill').style.left = `${50 + Math.min(1, Math.max(-1, v)) * 50 * (v >= 0 ? 0 : 1)}%`;
  $('mood-fill').style.width = `${Math.abs(v) * 50}%`;
  $('mood-fill').className = 'mood-fill ' + (v >= 0 ? 'pos' : 'neg');
  $('mood-text').textContent = `${v >= 0 ? '+' : ''}${v.toFixed(2)} for ${sideToMoveIsHuman ? 'you' : 'the fly'} (to move)`;
}

// ---------------------------------------------------------------- controls
function newGame() {
  state.chess = new Chess();
  state.selected = null; state.lastMove = null;
  state.human = $('color').value;
  state.depth = parseInt($('depth').value, 10);
  $('think-stats').textContent = '';
  renderBoard(); renderMoves(); updateBrainView();
  if (state.chess.turn() !== state.human) flyMove(); else setStatus('Your move.');
}
$('new').addEventListener('click', newGame);
$('color').addEventListener('change', newGame);
$('depth').addEventListener('change', () => { state.depth = parseInt($('depth').value, 10); });
$('undo').addEventListener('click', () => {
  if (state.thinking) return;
  state.chess.undo();
  if (state.chess.turn() !== state.human) state.chess.undo();
  state.selected = null; state.lastMove = null;
  renderBoard(); renderMoves(); updateBrainView();
  setStatus('Your move.');
});
$('hint').addEventListener('click', async () => {
  if (!state.ready || state.thinking || state.chess.turn() !== state.human || state.chess.isGameOver()) return;
  setStatus('The fly is thinking about your move…');
  const res = await ask({ type: 'move', fen: state.chess.fen(), depth: state.depth, history: positionKeys(), jitter: 0 });
  setStatus(`The fly would play ${res.move.san} (${(res.move.value >= 0 ? '+' : '') + res.move.value.toFixed(2)}). Your move.`);
});

function onReady() {
  drawLegend();
  const m = state.meta;
  const s = m.synapse_counts;
  console.log(`Fly brain ready: ${m.n_kc} KCs, ${m.n_mbon} MBONs, ${(s.pn_to_kc + s.kc_to_kc + s.kc_to_apl + s.apl_to_kc).toLocaleString()} fixed synapses, ${m.trained_synapses.toLocaleString()} trained KC→MBON synapses`);
  newGame();
}
renderBoard();

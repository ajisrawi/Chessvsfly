# 🪰 Chess vs Fly

Play chess against a fruit fly's brain.

The opponent is the **mushroom body** of the complete male *Drosophila* connectome
released by Google Research, HHMI Janelia's FlyEM team and the University of Cambridge
in October 2025 ([blog post](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)).
Every neuron and every synapse the fly thinks with is taken from that dataset; the only
thing we changed is the strength of the Kenyon-cell → MBON synapses, which is exactly the
set of synapses a real fly modifies when it learns.

* **Play:** https://ajisrawi.github.io/Chessvsfly/ (GitHub Pages), or open `web/index.html` locally by running
  `python -m http.server -d web` and visit http://localhost:8000.
* **Research notes:** [docs/RESEARCH.md](docs/RESEARCH.md) explains the connectome, the
  reconstruction pipeline and the biology behind each modelling choice.

## What is actually happening

```
chess position ─► 773 binary features ─► 773 projection neurons ─► 4,064 Kenyon cells ─► 97 MBONs ─► value
                 (piece on square,        (real neurons that       (real PN→KC, KC→KC,   (trained KC→MBON
                  castling, en passant)    innervate the calyx)     KC↔APL synapses)      synapses; valence
                                                                                         fixed by neurotransmitter)
```

1. **Presenting the board.** A position is converted to 773 features from the side to
   move's point of view. Each feature drives one of the 885 neurons that synapse onto
   Kenyon cells in the male CNS (olfactory, visual, thermo/hygrosensory projection
   neurons). Projection-neuron output is gain-controlled the way the antennal lobe does
   it, so an empty endgame drives the calyx as hard as a full board.
2. **The Kenyon-cell code.** The connectome's projection-neuron → Kenyon-cell synapse
   counts (43,590 connections), the KC → KC synapses (642,933 connections) and the
   KC ↔ APL feedback loop produce a sparse code across the 4,064 Kenyon cells, iterated for
   three time steps. Nothing here is learned.
3. **Learning.** Only the 61,210 Kenyon-cell → MBON synapses present in the connectome are
   plastic. They stay excitatory (weights are clipped at zero) and are fitted with Adam to
   Stockfish's evaluation of ~1M positions from noisy self-play, the engine standing in for
   dopamine.
4. **Deciding.** MBON valence is fixed by neurotransmitter as in Aso et al. 2014
   (glutamate → avoid, acetylcholine / GABA → approach). The fly's opinion of a position is
   `tanh(approach − avoidance)`. The fly plays the move leading to the position it likes
   best; at higher strengths it imagines your replies too (plain negamax over the fly's own
   evaluations, no other heuristics).

Chess rules and move generation come from [chess.js](https://github.com/jhlywa/chess.js);
they are the board, not the brain.

## Results

Training run committed in `data/fly_chess.npz` (600,000 positions, 30 epochs, ~10 min on 4 CPU cores):

| metric (held-out 18,000 positions) | value |
|---|---|
| correlation between fly value and `tanh(Stockfish cp / 500)` | **0.911** |
| agreement on which side is better | 88.8 % |
| mean squared error | 0.099 |
| plastic synapses (KC → MBON, all present in the connectome) | 61,040 |

Match play (Python engine, `flybrain.play`, 10 games each, colours alternating):

| fly | opponent | score |
|---|---|---|
| Fly (2 ply) | random mover | 10 – 0 |
| Larva (1 ply) | random mover | 6 – 4 (4 wins, 4 draws, 2 losses) |
| Fly (2 ply) | Stockfish 17.1 at UCI_Elo 1320, depth 1 | 1.5 – 8.5 (1 win, 1 draw, 8 losses) |

So the fly understands material and basic positional value well enough to crush a random
player and to win or draw the occasional game against a ~1300-rated engine, but it has no
tactical calculation beyond the 1–3 plies it imagines. Beginner-level chess from an insect,
which felt about right.

For comparison, a plain linear model on the raw 773 board features reaches a correlation of
0.86 on the same data; the fly's fixed sparse expansion plus its trained MBON layer beats
that. Design choices that mattered: assigning frequently occurring features to the
best-connected projection neurons (half of the 885 inputs contact fewer than 10 Kenyon
cells), distributing each projection neuron's output over its synapses in proportion to
synapse count, and scaling the initial KC → MBON weights so the output is not saturated
before learning starts.

## Reproduce

```bash
pip install -r requirements.txt
scripts/get_stockfish.sh && export STOCKFISH=$PWD/tools/stockfish/stockfish-ubuntu-x86-64-avx2

# 1. download the male CNS release tables (~570 MB) and cut out the mushroom body
python -m flybrain.connectome                     # -> data/circuit.npz

# 2. label positions with Stockfish (1M positions, ~10 min on 4 cores)
python flybrain/gen_positions.py --workers 4 --per-worker 250000

# 3. train the KC->MBON synapses
python -m flybrain.train --limit 600000 --epochs 25 --out data/fly_chess.npz

# 4. export for the browser and run the checks
python -m flybrain.export
python -m pytest -q

# 5. play in the terminal or measure strength
python -m flybrain.play --depth 2
python -m flybrain.play --vs random -n 20
python -m flybrain.play --vs stockfish --sf-elo 1320 --sf-depth 1 -n 10
```

`data/circuit.npz` (the extracted circuit) and `data/fly_chess.npz` (the trained synapses)
are committed, so steps 1–3 are only needed to retrain.

## Layout

| path | what |
|---|---|
| `flybrain/connectome.py` | download the release tables, extract the mushroom body circuit |
| `flybrain/encoder.py` | board → projection neurons → Kenyon cells (fixed) |
| `flybrain/model.py` | KC → MBON plastic synapses, valence readout, training rule |
| `flybrain/gen_positions.py` | Stockfish-labelled self-play positions |
| `flybrain/train.py` | training loop |
| `flybrain/export.py` | write `web/data/fly_brain.{json,bin}` |
| `flybrain/play.py` | play/evaluate from the terminal |
| `web/flybrain.js` | the same brain in JavaScript (active-neuron propagation) |
| `web/worker.js` | Web Worker: negamax over fly evaluations |
| `web/app.js`, `web/index.html`, `web/style.css` | the site, including a live view of Kenyon-cell and MBON activity |
| `tests/` | feature-encoding tests and Python ↔ JavaScript parity |
| `docs/RESEARCH.md` | research notes on the connectome and the modelling choices |

## Publishing the site

The workflow in `.github/workflows/pages.yml` deploys the `web/` folder to GitHub Pages on
every push to the default branch. Enable it once under *Settings → Pages → Source: GitHub
Actions*; the site is then served at `https://<user>.github.io/Chessvsfly/`. The site is
fully static (about 6 MB of synapses) and runs entirely in the visitor's browser.

## Credits

* Male CNS connectome: FlyEM Project Team (HHMI Janelia), Google Research Connectomics,
  Drosophila Connectomics Group (University of Cambridge / MRC LMB). Data CC-BY,
  `gs://flyem-male-cns`, https://male-cns.janelia.org/.
* Stockfish (labels only), chess.js (rules).

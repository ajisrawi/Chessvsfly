# Research notes: the male fruit fly connectome and how this project uses it

## 1. The milestone

On 22 October 2025 Google Research published *"A connectomics milestone: mapping the
complete male fruit fly brain"* together with the FlyEM project team at HHMI Janelia
Research Campus and the Drosophila Connectomics Group at the University of Cambridge
(with the MRC Laboratory of Molecular Biology). The release, called the **Male CNS**
connectome, is the first synapse-resolution wiring diagram of an *entire* central
nervous system of an adult male *Drosophila melanogaster*: the brain, both optic
lobes and the ventral nerve cord (the fly's "spinal cord") of one animal, in a single
seamless dataset.

Headline numbers (from the Janelia / HHMI announcements and the release bucket):

| quantity | value |
|---|---|
| neurons (traced) | ~166,000 (brain + ventral nerve cord) |
| synaptic connections | ~125 million |
| neuron-to-neuron edges (traced-only weight table) | 25.6 million |
| EM resolution | 8 nm isotropic |
| licence | CC-BY |

The flagship paper and a companion on sexual dimorphism appeared as a Cell consortium
("Drosophila Male Fly Connectome"); comparing the male map with the female FlyWire
brain identified 262 sex-specific and 114 sexually dimorphic cell types (about 4.8 % of
the central brain). Data version v0.9 shipped with the announcement; v1.0 (used here)
was released in June 2026.

## 2. What "the model" is

The blog post's contribution is not a single trained network but a reconstruction
pipeline plus its output:

* **Imaging** at Janelia: the whole CNS was cut with a hot-knife and imaged with
  enhanced focused-ion-beam scanning electron microscopy at 8 nm.
* **Alignment and segmentation** at Google Research: the image volume was aligned and
  every neuron was traced with **flood-filling networks** (FFN, Januszewski et al.
  2018), the recurrent CNN that grows one object at a time from a seed. The male CNS
  also used the newer *PATHFINDER* agglomeration system, trained partly on synthetic
  neurons, to merge FFN fragments into whole cells with far fewer proofreading hours.
* **Synapses** were detected automatically and each presynaptic site was assigned a
  predicted **neurotransmitter** (acetylcholine, GABA, glutamate, dopamine,
  serotonin, octopamine) using the classifier approach of Eckstein et al. 2024.
* **Proofreading and annotation** at Janelia and Cambridge produced cell types for
  every neuron; the data are served through neuPrint, Neuroglancer and flat files in
  the public bucket `gs://flyem-male-cns`.

So the "model" is the connectome itself: a graph whose nodes are neurons and whose
weighted edges are synapse counts, with a predicted sign (excitatory/inhibitory) per
neuron. That graph is what we run.

## 3. Files we use

From `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`:

| file | content | size |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | cell type, class, side for 211,577 bodies | 14 MB |
| `body-neurotransmitters-male-cns-v1.0.feather` | consensus neurotransmitter per neuron | 43 MB |
| `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` | 25.6 M `(pre, post, synapse count)` edges | 508 MB |

`flybrain/connectome.py` downloads these and extracts the circuit below.

## 4. The circuit: the mushroom body

We chose the **mushroom body**, the fly's associative-learning centre, because it is
the part of the fly that *learns*:

```
 projection neurons  ───►  Kenyon cells (4,064)  ───►  MBONs (97)  ───►  approach / avoid
 (885 inputs)               ▲   │         │
                            │   ▼         ▼  (plastic synapses, dopamine-gated)
                           APL (2, GABA)  ◄─┘
```

Numbers in the male CNS v1.0 (all synapse counts are real):

| pathway | edges | synapses |
|---|---|---|
| inputs → Kenyon cells | 43,590 | 464,277 |
| Kenyon cell → Kenyon cell | 642,933 | 1,153,845 |
| Kenyon cell → APL | 4,693 | 210,352 |
| APL → Kenyon cell | 4,633 | 196,200 |
| Kenyon cell → MBON | 61,210 | 463,640 |
| MBON → MBON | 1,606 | 26,259 |

Kenyon-cell subtypes present: γ (KCg-*), α/β (KCab-*) and α'/β' (KCa'b'-*). All
Kenyon cells are cholinergic; the two APL neurons are GABAergic; MBONs are 50
cholinergic, 21 GABAergic and 26 glutamatergic. Following Aso et al. 2014 (*eLife*
3:e04580), glutamatergic MBONs are treated as driving avoidance and the others as
driving approach; that is the only "output wiring" we add, and it is fixed.

## 5. Reading a chessboard into the fly

The fly has no eyes in this model. Instead, as in the browser games built on the
female FlyWire connectome, we stimulate neurons directly. A position is turned into
773 binary features (64 squares × 12 piece kinds, from the side to move's point of
view, plus 4 castling rights and an en-passant flag). Each feature is presented as
activity in the projection neurons that innervate the calyx, using the
synapse-count wiring from the connectome to drive Kenyon cells. Two pieces of real
fly physiology are kept:

* **Antennal-lobe gain control** (Olsen & Wilson 2010): total projection-neuron output
  is normalised so that a sparse endgame drives the calyx as strongly as a crowded
  middlegame.
* **APL feedback inhibition** (Lin et al. 2014): the APL neuron sums Kenyon-cell
  activity and inhibits all Kenyon cells, keeping the code sparse.

The Kenyon-cell code is iterated for three time steps including the KC→KC
recurrence. Nothing in this front end is trained.

## 6. Learning

In the fly, learning is dopamine-gated plasticity of the excitatory Kenyon-cell→MBON
synapses (Hige et al. 2015). We keep exactly that: only the 61,210 KC→MBON synapses
that exist in the connectome are modifiable, they are never allowed to change sign,
and every other weight is frozen. Stockfish's evaluation of a position is the
"dopamine": the fly's output (approach minus avoidance MBON activity, squashed with
tanh) is regressed onto `tanh(centipawns / 500)` over roughly a million positions from
noisy self-play games.

## 7. Honest limits

* The male CNS has no photoreceptors in this model; the board is injected into
  projection neurons, so this is "seeing" only by analogy.
* The dynamics are a rate model with three discrete steps, not spiking neurons.
* Mapping board features to specific projection neurons is our (deterministic)
  choice; the connectome does not tell you which neuron should smell a knight.
* Chess rules and the look-ahead are ordinary code; the fly is the position
  evaluator only.

## 8. Sources

* Google Research blog (22 Oct 2025): https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/
* Janelia FlyEM Male CNS project: https://www.janelia.org/project-team/flyem/male-cns-connectome
* Male CNS data portal and downloads: https://male-cns.janelia.org/ ; bucket `gs://flyem-male-cns`
* Cell consortium page: https://www.cell.com/consortium/male-fly-connectome
* HHMI news: https://www.hhmi.org/news/scientists-complete-full-map-fruit-fly-brain-connectome
* Januszewski et al. 2018, *Nature Methods*, "High-precision automated reconstruction of neurons with flood-filling networks"
* Aso et al. 2014, *eLife*, "Mushroom body output neurons encode valence and guide memory-based action selection"
* Lin et al. 2014, *Nature Neuroscience*, "Sparse, decorrelated odor coding in the mushroom body enhances learned odor discrimination"
* Hige et al. 2015, *Neuron*, "Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila"
* Olsen & Wilson 2010, *Neuron*, "Divisive normalization in olfactory population codes"
* Eckstein et al. 2024, *Cell*, "Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster"
* Shiu et al. 2024, *Nature*, "A Drosophila computational brain model reveals sensorimotor processing" (whole-brain leaky integrate-and-fire modelling that inspired the rate model here)

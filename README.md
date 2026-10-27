# sr-wordclasses

Code for *Word-Class and Construction-Like Structure Emerges in Neural Successor
Representations Trained on Natural Language* (Immertreu, Schilling, Kinfe & Krauss,
preprint [arXiv:2605.24585](https://arxiv.org/abs/2605.24585)).

A residual network is trained on WikiText-103 to predict successor representations
(SRs) at three horizons (gamma = 0.2, 0.5, 0.8). The per-word SR vectors are then
clustered without part-of-speech supervision and compared post hoc with reference tags.
This repository holds the pipeline, reduced to the path the paper reports, and the
tests that show how it was checked.

The trained model and every intermediate result are in the data deposit
(Zenodo, DOI [10.5281/zenodo.22256000](https://doi.org/10.5281/zenodo.22256000)).
Point `SRWC_DEPOSIT` at the unpacked deposit, the folder that contains `reproduce.py`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.12
pip install --no-deps -r requirements.txt && pip install --no-deps -e .
pip install --no-deps -r requirements-figures.txt      # only for scripts/figures
export SRWC_DEPOSIT=/path/to/zenodo_wordclass_sr
pytest tests/audit                                      # about a minute
```

`requirements.txt` is a complete lock of the environment the checks ran in. Install it
with `--no-deps`: that environment ran numpy 1.26.4 under thinc 8.3.6, which declares
numpy >= 2, so pip's resolver would refuse the set. This exact install was tested from
scratch, and all audit and regeneration tests pass in it. WikiText-103 is downloaded
through Hugging Face `datasets` on first use. The NLTK Penn Treebank sample,
used for 18 POS fallbacks, downloads automatically. The figure scripts additionally need
`requirements-figures.txt` (matplotlib, seaborn, umap-learn), locked to the versions the
paper's figures were drawn with.

## Pipeline

| Step | Code | Paper |
|---|---|---|
| Vocabulary (20,000 types) | `srwc/vocab.py` | Methods: Training |
| Corpus preprocessing | `srwc/corpus.py` | Methods: Training |
| Network | `srwc/model.py` | Appendix E.1 |
| TD(lambda) targets and KL loss | `srwc/targets.py` | Methods; Appendix E.2 |
| Training | `srwc/train.py`, `scripts/01_train.py` | Methods, Table 1 |
| Reference POS tags and evaluated words | `srwc/lexicon.py`, `srwc/select.py`, `scripts/02_extract_sr.py` | Methods, Table 2 |
| Consensus clustering | `srwc/cluster.py`, `scripts/03_cluster.py` | Methods; Appendix E.3 |
| Purity, NMI, ARI scans | `srwc/metrics.py`, `scripts/04_curves.py` | Figures 7, 10; Appendix E.4 |
| Inter-cluster transitions | `srwc/metrics.py`, `scripts/05_transitions.py` | Figures 5, 9; Appendix E.5 |
| Cluster listings | `scripts/06_appendix_listings.py` | Appendices A-D |
| Target composition, padding share | `scripts/appendix_E2.py` | Appendix E.2 |
| Reserved vocabulary dimensions | `scripts/appendix_F1.py` | Appendix F.1 |
| Robustness grid and its tables | `scripts/appendix_F.py`, `scripts/appendix_F_tables.py` | Appendices F.2-F.4 |
| Re-training stability | `scripts/appendix_F3.py` | Appendix F.3, Table S2 |
| Feature scale | `scripts/appendix_F5.py` | Appendix F.5 |
| Count-based controls | `srwc/baselines.py`, `scripts/appendix_G.py` | Appendix G |
| Figures 1-10 | `scripts/figures/fig*.py` (data via `figdata.py`) | Figures 1-10 |

All figures are drawn from the deposit and show the same data as those of the paper from
arXiv v4 on. Rendering can differ in layout: on our machine, Figures 3 and 9 come out with
slightly different spacing and, in Figure 9, a different placement of the network nodes. The
UMAP panels depend on the numerical environment; the untrained panels of Figure 4 change even
with the thread settings (e.g. `OMP_NUM_THREADS`). Two figures differ from earlier versions,
and their scripts explain why: in Figures 4b and 6a, earlier versions placed words at UMAP
coordinates computed in another table's row order, and the untrained panels of Figure 4 now
come from a seeded initialisation, because the original was unseeded.

Each module's docstring states what it does in the reported run. Some details there
are easy to miss:

* **Corpus.** Each WikiText-103 line is one sample, cut to its first 80 tokens, and
  every line begins with `<unk>`, a side effect of the two-pass preprocessing that is
  kept as run.
* **Learning rate.** The cosine schedule decays to zero.
* **POS lexicon.** The reference tags come from `en_core_web_lg` over the first 2,000
  lines of the training split, with an NLTK fallback for words missing there.
* **Count-based controls.** These are counted on the model's own preprocessed training
  tokens. Appendix G reports means over four consensus seeds (`appendix_G.py seeds`),
  because at NVA K=3 the count features switch between distinct partitions from seed to
  seed. `--tokens single-pass` reproduces the single-pass tokenisation of the first
  version of the paper.

## What can be verified, and how

### 1. Every reported number, exactly, on any machine (`tests/audit`)

Starting from the deposited files, the audit tests recompute the following and require
the deposited values (exactly, or to float tolerance where floats are involved):

* the 62 headline values checked by the deposit's own `reproduce.py`;
* the vocabulary;
* the six evaluated word tables;
* the SR vectors, recomputed from the checkpoint;
* the 1,800-row quality scan;
* both transition matrices;
* the Appendix A-D listings, checked against the supplement's text when
  `SRWC_SUPPLEMENT` points to it.

In two-word clusters the listing order is an exact tie, which the original run broke
by rounding noise; the test accepts either order.

### 2. Regenerating results: equivalent, not identical (`tests/regeneration`, `scripts/check_regeneration.py`)

Training, PCA and the consensus clustering run in floating point. Their results
depend on the CPU, the BLAS and OpenMP libraries, the thread count and library
versions. The consensus clustering is especially sensitive:
* co-association values are multiples of 1/240, so average linkage meets near-ties;
* one base partition that flips between two near-equal solutions can reorder merges.

The deposited clusterings were computed on a compute cluster. On any other machine,
expect partitions that differ from the deposited ones roughly as much as re-seeding
the consensus aggregation does (Supplementary Appendix F.3), while agreement with the
reference tags, which is what the paper reports, stays essentially unchanged.
`scripts/check_regeneration.py` measures both for a set of regenerated clusterings.

Evidence from one other machine (a 24-core workstation, 8 threads) against the deposit:

The regeneration covered 10 of the 12 scans: the six KMeans scans and the four BGM scans
that did not already match exactly. The table shows the showcased resolutions:

| | identical to deposit (ARI 1.000) | ARI vs deposit otherwise | change in agreement with tags |
|---|---|---|---|
| 20 scan x resolution cells | 10 | 0.714-0.995 | ARI <= 0.019, NMI <= 0.018, mean purity <= 0.014 |

Most differing cells lie within or above the re-seeding range of Appendix F.3: 0.867-0.945
at the coarse and 0.744-0.871 at the fine resolutions. One does not: BGM, NVA, gamma = 0.8,
K = 3 agrees with the deposit at 0.714, with agreement with the tags 0.019 ARI higher.
Partitions can therefore move further than re-seeding moves them, while the scores the
paper reports stay within 0.02.

Within one machine, repeated runs are identical: PCA and the base partitions run with
a fixed thread count (8; override with `SRWC_THREADS`).

Retraining the network takes about 15 hours on one NVIDIA A40 (the reported model: 10 epochs in 15 h 11 min). The reported model was trained
without a fixed seed, so retraining gives a new model rather than the same weights. Its
results should agree with the paper's within the variation between the two retrained
models of Appendix F.3.

### 3. Equivalence to the original research code (`tests/equivalence`)

This repository is a clean reimplementation of research code that is not public. Each
module was checked against that code on the same machine with identical settings. Token
IDs, training losses and gradients, the clustering and the count baselines came out
identical, apart from float rounding (training weights within 1e-6 after two epochs;
analytic SR features within 3e-8). The tests need the original code (`SRWC_ORIGINAL`) and
are skipped otherwise; they are kept to document the validation.

## Licence

MIT, see `LICENSE`. Please cite the article (`CITATION.cff`).

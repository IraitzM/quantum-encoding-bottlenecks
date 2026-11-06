# How Quantum Data Encoding Constrains Representation Learning: code

This repository reproduces every number of the article, from the pooling screens
to the regime map. It simulates angle encoding with data re-uploading, amplitude
embedding and the quantum spectral model (QSM) on five classification tasks, and
measures where accuracy is lost along the pipeline: in pooling, in the function
class of the encoding, or in training. Every simulation is noiseless and uses
exact expectation values.

The repository stops at the numbers. The figures of the article are drawn
elsewhere from the CSVs that `analysis/tables.py` writes, and no plotting code
is included.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/IraitzM/qenc
cd qenc
uv sync
```

`uv.lock` pins the library versions that produced the published results
(PyTorch 2.12.1, PennyLane 0.45.0, scikit-learn 1.9.0, NumPy 2.4.6).
Everything runs on a CPU. Only the regime map benefits from a GPU.

## Data

Every dataset is downloaded on first use into `data/` and checked against its
MD5 checksum. Set `QENC_DATA` to use another folder.

| Task | Source |
|---|---|
| ECG | MIT-BIH Arrhythmia Database 1.0.0, PhysioNet (records 106, 119, 200, 208 for training, 203 and 223 for testing), read with `wfdb` |
| EEG | Bonn recordings in the UCI Epileptic Seizure Recognition release (Hugging Face mirror) |
| MNIST1D | official release (Greydanus and Wang, 2024) |
| PneumoniaMNIST, NoduleMNIST3D | MedMNIST, Zenodo record 10519652 (28-pixel archives, plus the 64 and 224 MedMNIST+ archives for the regime map) |
| HLS-CMDS | heart and lung recordings, Zenodo record 15376628 |

PhysioNet can be slow. At the 5 kB/s we measured, the six MIT-BIH records take about a quarter of an hour to
download on first use.

## Layout

```
src/qenc/       the library
  data.py         downloads and the data draw of every experiment
  pooling.py      block averaging A, its projection A^+A and the energy it discards
  models.py       the three encodings as PennyLane classifiers, and the QSM uploads
  protocols.py    draws, candidates and inputs of the five-task study and the spatial pilot
  measures.py     final states, Pauli and readout features, kernels, and the classical fits
  rank.py         two-qubit entangler ladder for the controlled-rank regressions
  fastsim/        batched torch statevector simulator used by the regime map
experiments/    one script per result bundle, with the paper's settings fixed inside
analysis/       tables.py, every number of the paper from the per-job results
results/        written by the runs, one folder per bundle (created on first use)
protocols/      the frozen protocols of its experiments, named by the identifiers the article cites
```

Results go to `results/<bundle>/<job>/` (set `QENC_RESULTS` to change the folder) and
the summaries to `results/summary/`. Each experiment takes `--list`, `--job NAME`
(repeatable) or `--all`. A finished job is skipped and its results are never
overwritten, so an interrupted run resumes.

## Reproducing the numbers

Run the experiments in this order, since the probes and the ladder analyze the
models saved by the first three, then the analysis on the same folder:

```bash
uv run experiments/five_task.py --all
uv run experiments/spatial_pilot.py --all
uv run experiments/volume_followup.py --all
uv run experiments/detail.py --all
uv run experiments/probes.py --all
uv run experiments/ladder.py --all
uv run experiments/planted.py --all
uv run experiments/regime_map.py --all
uv run experiments/regime_map.py --ntk-norm
uv run experiments/witnesses.py --all
uv run experiments/eeg_controls.py --all
uv run experiments/hls.py --all
uv run experiments/rank_restarts.py --all
uv run analysis/tables.py
```

Jobs are independent, so `--job` lets you spread an experiment over several processes or
machines. `analysis/tables.py` skips bundles that are missing, so a partial rerun still
summarizes what it has.

| Article | Experiment | Summary in `results/summary/` | Time per job on one core |
|---|---|---|---|
| Ladder figure, gaps of Sec. 3.1 to 3.3 | `ladder.py` (25 jobs) | `ladder_gaps.csv`, `ladder_levels.csv`, `tex/ladder_table.tex` | about 10 minutes |
| Witnesses of Sec. 3.1 | `witnesses.py` | `witness_phase.csv`, `witness_label_rule.csv` | seconds |
| Pooling screen $G_A$ (Eq. 3) | `five_task.py` (25 jobs) | `five_task_pooling_screen.csv` | 5 to 30 minutes |
| Retained detail at fixed capacity | `detail.py` (10 jobs) | `detail_contrasts.csv` | about 1 minute |
| EEG and audio preprocessing controls | `eeg_controls.py` (5 jobs), `hls.py` (10 jobs) | `eeg_controls.csv`, `hls.csv` | about 12 and 5 minutes |
| Planted labels F2 and F4, frequency labels | `planted.py` (70 jobs) | `planted_summary.csv`, `planted_contrasts.csv` | 5 to 10 minutes |
| Restart statistics of the controlled regressions | `rank_restarts.py` | `rank_restarts.csv` | 7680 fits of 20 seconds, run in parallel |
| Feature probes, initial against trained | `probes.py` (25 jobs) | `probe_contrasts.csv`, `tex/probe_table.tex`, `tex/resource_table.tex` | a few minutes |
| Regime map | `regime_map.py` (320 jobs) | `regime_verdicts.json`, `regime_summary.csv`, `tex/regime_*.tex` | minutes on a GPU; the 10-qubit QSM and 16-qubit jobs take hours and about 24 GB |
| Validation selection | `five_task.py`, `volume_followup.py` (5 jobs) | `validation_selection.csv`, `volume_followup.csv` | see above; 3 minutes |
| Every quoted number and table | `analysis/tables.py` | `results/summary/*.csv` and `tex/` | a few minutes |

With the pinned library versions, reruns of sampled jobs of every experiment
reproduced the published scores and predictions exactly, on CPU and, for the
regime map, on the same GPU model. Floating-point differences between machines
or thread counts can still flip an occasional prediction that sits on a decision
boundary.

## Protocols

`protocols/` holds the frozen protocols byte for byte, so the `protocol_sha256`
recorded in the jobs verifies. Besides the design and the predictions, a
protocol lists the hashes of the development code that ran it, which is not part
of this repository.

| Protocol | Script |
|---|---|
| `geometry-validation-v2` | `five_task.py` |
| `volume-encoding-followup-v1` | `volume_followup.py` |
| `witnesses` | `witnesses.py` |
| `spatial-qsm-v1` | `spatial_pilot.py` |
| `representation-bottlenecks-resolution-v1` | `detail.py` |
| `retrospective-three-family-probes-v1` | `probes.py` |
| `ladder-pilot-v1` | `ladder.py` |
| `planted-labels-v2` | `planted.py` |
| `programme-phase1-v1` | `regime_map.py` |

## Citation

If you use this code, please cite the article. `CITATION.cff` holds the reference, which will be completed once
the article is published.

## Licence

MIT, see `LICENSE`.

## Acknowledgements

The checkerboard generator in `qenc.data.checkerboard` is adapted from
[kernel-trainer](https://github.com/IraitzM/kernel-trainer) (MIT license).

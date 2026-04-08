# RADD: Socially-Aware Trajectory Anomaly Detection via Dual-View Fusion

This repository contains the code and paper for **RADD** (Robustness-Aware Dual-view Detection), a self-supervised dual-view framework for trajectory anomaly detection in Location-Based Social Networks (LBSNs).

## Overview

RADD scores trajectory anomalies from two complementary views:
- **Personal view**: deviation from the user's own historical mobility prototype
- **Social view**: deviation from a prototype aggregated over the user's top-*k* mobility-relevant friends

The two calibrated scores are fused via simple weighted combination, requiring no anomaly labels.

### Key Results

| Dataset | GRU-AE | RADD | Delta |
|---------|--------|------|-------|
| Brightkite (sparse graph) | **.752** | .726 | -2.6 |
| Gowalla (dense graph) | .661 | **.714** | **+5.3** |

Social augmentation benefit depends on dataset-level social signal quality: RADD improves substantially on Gowalla (dense social graph) but not on Brightkite (sparse social graph).

## Project Structure

```
RADD/
├── src/
│   ├── model.py                    # RADD model (GRU-AE, LSTM-AE, RADD, EarlyFusionAE)
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── anomaly_injection.py        # Synthetic anomaly injection (4 types)
│   ├── run_experiments.py          # Main experiment runner
│   └── run_traditional_baselines.py # IF, LOF, OCSVM baselines
├── data/                           # Place datasets here (see below)
├── results/                        # Experiment results (JSON)
├── paper/
│   ├── main.tex                    # Paper source (IEEE conference format)
│   ├── references.bib              # Bibliography
│   └── main.pdf                    # Compiled paper
├── requirements.txt
└── README.md
```

## Setup

### Environment

```bash
conda create -n radd python=3.9
conda activate radd
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA following [pytorch.org](https://pytorch.org/get-started/locally/).

### Data

Download Brightkite and Gowalla datasets from [SNAP Stanford](https://snap.stanford.edu/data/):

- [Brightkite check-ins](https://snap.stanford.edu/data/loc-Brightkite.html)
- [Gowalla check-ins](https://snap.stanford.edu/data/loc-Gowalla.html)

Place the gzipped files in the `data/` directory:
```
data/
├── brightkite_checkins.txt.gz
├── brightkite_edges.txt.gz
├── gowalla_checkins.txt.gz
└── gowalla_edges.txt.gz
```

## Usage

### Run All Experiments

```bash
cd src

# Brightkite
python run_experiments.py \
    --dataset brightkite \
    --data_dir ../data \
    --results_dir ../results \
    --seeds 42 123 456 \
    --blocks block1 block6

# Gowalla
python run_experiments.py \
    --dataset gowalla \
    --data_dir ../data \
    --results_dir ../results \
    --seeds 42 123 456 \
    --blocks block1 block6
```

### Experiment Blocks

| Block | Description |
|-------|-------------|
| `block1` | Main results: all baselines + RADD + ablations (3 seeds) |
| `block2` | Ablation study |
| `block3` | User activity group analysis |
| `block4` | Social robustness (edge dropout 20%-80%) |
| `block5` | Top-k sensitivity |
| `block6` | Friend-count group analysis |

### Run Traditional Baselines

```bash
python run_traditional_baselines.py \
    --dataset brightkite \
    --data_dir ../data \
    --results_dir ../results
```

## Method

1. **Shared Trajectory Encoder**: GRU autoencoder trained on normal check-in sequences with cross-entropy reconstruction loss
2. **Personal Branch**: Cosine distance between trajectory embedding and user's historical prototype
3. **Social Branch**: Cosine distance between trajectory embedding and top-*k* friends' aggregated prototype (weighted by POI co-visitation similarity)
4. **Fusion**: Z-score normalization + weighted combination with fixed alpha=0.5

## Anomaly Types

| Type | Description |
|------|-------------|
| POI Replacement | Replace 1-3 POIs with random globally popular POIs |
| Time Shift | Shift trajectory to an unusual time window |
| Subtrajectory Splice | Replace a segment with another user's segment |
| Social Inconsistency | Replace POIs with locations far from user and friends |


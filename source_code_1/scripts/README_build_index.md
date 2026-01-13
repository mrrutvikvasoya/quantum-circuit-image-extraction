# Building Reference Index - Guide

## Overview

This script builds the FAISS reference index used for circuit detection from labeled images.

## Prerequisites

**1. Labeled Images Directory Structure:**
```
data/labeled_images/
├── circuits/           # Quantum circuit images
│   ├── circuit_001.png
│   ├── circuit_002.png
│   └── ...
└── non_circuits/       # Non-circuit images (graphs, plots, etc.)
    ├── graph_001.png
    ├── plot_002.png
    └── ...
```

**2. Required:**
- At least 100+ circuit images
- At least 100+ non-circuit images
- Images should be representative samples

## Running the Script

```bash
# From project root
python scripts/build_reference_index.py
```

## What it Does

### Step 1: Load Images
- Scans `data/labeled_images/circuits/`
- Scans `data/labeled_images/non_circuits/`
- Reports count of images found

### Step 2: Initialize DINOv2
- Loads pre-trained `facebook/dinov2-small` model
- Sets up GPU/CPU

### Step 3: Generate Embeddings
- Processes each image through DINOv2
- Extracts 384-dimensional embedding vector
- Shows progress bar

### Step 4: Compute Centroids
- Calculates average (centroid) of all circuit embeddings
- Calculates average (centroid) of all non-circuit embeddings

### Step 5: Build FAISS Index
- Combines all embeddings into single index
- Creates L2 (Euclidean distance) index
- Enables fast K-nearest neighbor search

### Step 6: Save to Disk
- Saves to `models/dinov2_index/`
- Creates 4 files (see below)

## Output Files

```
models/dinov2_index/
├── reference_embeddings.pkl    # All embeddings + image paths
├── cluster_centroids.pkl       # Circuit & non-circuit centroids
├── reference_index.faiss       # FAISS index for similarity search
└── index_metadata.pkl          # Counts and dimensions
```

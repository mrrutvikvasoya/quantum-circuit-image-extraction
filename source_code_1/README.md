# ProjectNLP: Quantum Circuit Extraction Pipeline

Automated extraction of quantum circuit images from arXiv papers using deep learning.

Name: Rutvik Dilipbhai Vasoya
Id: 1
Enrollment no: 34925886

## Overview

This pipeline:
1. Downloads quantum physics papers from arXiv
2. Extracts images using multi-method PDF processing
3. Detects circuits via DINOv2 embeddings + FAISS similarity
4. Validates with visual gate detection (EasyOCR)
5. Classifies problem types using SciBERT
6. Outputs structured JSON dataset with metadata

## Installation

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

```

## Usage

```bash
python src/main.py
```

**Input**: Text file with arXiv IDs (one per line) in `input/paper_list_1.txt`

**Output**:
- `output/dataset_X.json` - Structured circuit metadata
- `output/circuits_X/` - Extracted circuit images
- `output/paper_list_counts_X.csv` - Processing summary

## Architecture

```
InputHandler -> DownloadManager -> PdfImageExtractor
                                        |
                                        v
                            DetectionOrchestrator
                            [CaptionFilter -> DINOv2 -> GateDetector -> SciBERT]
                                        |
                                        v
                MetadataCompiler -> CheckpointManager -> OutputGenerator
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `DINOv2` | Image embeddings for circuit classification |
| `SciBERT` | Problem classification (Grover, Shor, VQE, etc.) |
| `EasyOCR` | Visual gate detection (H, X, CNOT, CZ...) |
| `FAISS` | Fast similarity search on embeddings |
| `CaptionFilter` | Whitelist/blacklist pre-screening |

## Configuration

Edit `config/config.yaml`:

```yaml
pipeline:
  maxCircuitCount: 250     # Stop after N circuits
  minImageSize: 150        # Min image dimension


detection:
  embedding:
    combinedThreshold: 0.7   # Detection threshold
    highConfidence: 0.80     # HIGH tier
    mediumConfidence: 0.60   # MEDIUM tier
```

## Project Structure

```
ProjectNLP/
  config/config.yaml         # Settings
  input/paper_list_1.txt     # arXiv IDs
  output/                    # Results
  models/dinov2_index/       # Pre-built index
  scripts/                   # Utilities
  src/
    main.py                  # Entry point
    detectors/               # Detection modules
    *.py                     # Pipeline modules
```

## Module Descriptions

### Core Pipeline (`src/`)

| File | Description |
|------|-------------|
| `main.py` | Entry point. Parses arguments, initializes all modules, runs the main processing loop over papers. |
| `utils.py` | Utility functions and data classes (`ConfigLoader`, `BoundingBox`, `ImageInfo`, `PdfData`, `DetectionResult`, `CompleteMetadata`). |
| `input_handler.py` | Reads and validates the arXiv paper list. Cleans IDs, checks formats, removes duplicates. |
| `download_manager.py` | Downloads PDFs from arXiv API. Validates paper category (quant-ph), retrieves title/abstract. |
| `pdf_image_extractor.py` | Extracts images from PDFs using 3 methods (embedded, vector, rendered). Parses captions, figure numbers, surrounding text. |
| `metadata_compiler.py` | Combines detection results with PDF metadata. Selects best description, normalizes gates, builds final record. |
| `checkpoint_manager.py` | Manages pipeline checkpoints for resume capability. Saves/loads progress atomically. |
| `output_generator.py` | Generates all output files: JSON dataset, CSV tracking, circuit images. Uses atomic writes. |

### Detectors (`src/detectors/`)

| File | Description |
|------|-------------|
| `detection_orchestrator.py` | Coordinates multi-stage detection: caption filter → DINOv2 → gate detector → problem classifier. |
| `embedding_detector.py` | DINOv2-based circuit classifier. Uses FAISS for similarity search and centroid comparison. |
| `embedding_encoder.py` | Wraps DINOv2 model. Handles image preprocessing, embedding generation, L2 normalization. |
| `embedding_config.py` | Configuration constants for DINOv2 (model name, thresholds, file paths). |
| `caption_filter.py` | Pre-filters images using caption keywords. Returns ACCEPT/REJECT/PASS decision. |
| `visual_gate_detector.py` | Detects quantum gates using contour analysis + EasyOCR. Validates against known gate vocabulary. |
| `matrix_detector.py` | Identifies matrix representations to filter out non-circuit images (brackets, grid patterns). |
| `quantum_problem_classifier.py` | Classifies problem type (Grover, Shor, VQE, etc.) using phrase matching + SciBERT embeddings. |
| `quantum_problem_config.py` | Reference phrases and texts for each quantum problem category. |

### Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `build_reference_index.py` | Builds DINOv2 reference index. Generates embeddings, computes centroids, creates FAISS index. |
| `Dataset_Preparation.py` | Utility for preparing training datasets. Randomly samples and converts images. |

## Dependencies

- PyTorch, Transformers, timm (DINOv2)
- PyMuPDF, docling (PDF processing)
- EasyOCR, pytesseract (OCR)
- FAISS (vector search)
- arxiv (API client)

----

*NLP Course Project - Quantum Circuit Extraction*
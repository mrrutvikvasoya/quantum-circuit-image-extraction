# 🔬 Quantum Circuit Image-to-Text Dataset Pipeline

Automated pipeline for extracting and compiling quantum circuit diagrams from scientific papers with metadata for training image-to-text models.

## 🎯 Project Overview

This pipeline automatically extracts quantum circuit diagrams from arXiv papers (quant-ph category) and generates a structured dataset with descriptive metadata. Standard image captioning models fail on schematic images like quantum circuits - this dataset addresses that gap.

**Key Achievement:** Compiled 250 quantum circuit images with 94% metadata completeness in 79 minutes, achieving 80% precision.

## ✨ Key Features

- **Multi-Method Extraction** - Captures embedded, vector, and rendered circuits from PDFs
- **3-Stage Detection** - Caption filtering → DINOv2 visual embeddings → OCR gate validation
- **Problem Classification** - Identifies 13 quantum algorithm categories using SciBERT
- **Rich Metadata** - Includes gates, problem types, descriptions, and character-level positions
- **Fully Automated** - Processes papers sequentially with checkpointing

## 🏗️ Pipeline Architecture

```
PDF Download → Image Extraction → Detection & Filtering → 
Problem Classification → Metadata Compilation → JSON Output
```

### Detection Pipeline

1. **Caption-Based Pre-filtering** - Keyword whitelist/blacklist (filters 11.3% immediately)
2. **DINOv2 Visual Embeddings** - Dual classification (centroid + k-NN) with 0.70 threshold
3. **OCR Gate Validation** - EasyOCR with preprocessing (CLAHE + adaptive threshold)

### Extraction Methods

- **Embedded Images** - Direct PNG/JPEG extraction from PDF
- **Vector Graphics** - LaTeX/TikZ circuits via spatial clustering
- **Rendered Regions** - Hybrid circuits with morphological operations

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Papers Processed | 1,041 |
| Processing Time | 79 minutes (~4.6 sec/paper) |
| Images Extracted | 8,772 |
| Circuits Detected | 250 |
| Precision | 80% (200 correct) |
| Metadata Completeness | 94% |

## 🛠️ Tech Stack

- **Python 3.x** - Core language
- **PyMuPDF** - PDF processing and image extraction
- **DINOv2** - Self-supervised visual embeddings (facebook/dinov2-small)
- **SciBERT** - Scientific text embeddings for problem classification
- **EasyOCR** - Gate symbol detection
- **FAISS** - Efficient similarity search
- **OpenCV** - Image preprocessing and contour detection

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/quantum-circuit-dataset.git
cd quantum-circuit-dataset

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py --config config.yaml
```

## 📁 Project Structure

```
├── src/
│   ├── main.py                      # Pipeline orchestrator
│   ├── download_manager.py          # arXiv download & validation
│   ├── pdf_image_extractor.py       # Multi-method extraction
│   ├── detection_orchestrator.py    # Detection coordination
│   ├── caption_filter.py            # Keyword filtering
│   ├── embedding_detector.py        # DINOv2 classification
│   ├── visual_gate_detector.py      # OCR validation
│   ├── quantum_problem_classifier.py # SciBERT classification
│   ├── metadata_compiler.py         # Metadata extraction
│   └── checkpoint_manager.py        # Progress tracking
├── config/
│   ├── config.yaml                  # Pipeline configuration
│   └── quantum_problem_config.py    # Problem categories
├── data/
│   ├── reference_embeddings/        # DINOv2 reference database
│   └── output/                      # Generated dataset
└── requirements.txt
```

## 📦 Output Format

```json
{
  "arxiv_number": "2504.13910",
  "page_number": 5,
  "figure_number": "3",
  "quantum_gates": ["H", "CNOT", "Rx", "Rz"],
  "quantum_problem": "QAOA",
  "descriptions": "Circuit implementing QAOA for MaxCut...",
  "text_positions": [[1250, 1450]]
}
```

## 🔬 Methodology Highlights

### Visual Classification Strategy

Combines two complementary methods:
- **Centroid-based** (60% weight) - Global class structure
- **k-NN** (40% weight) - Local similarity with k=10

Combined score threshold: 0.70 (optimized for precision-efficiency balance)

### Problem Classification

Two-stage approach:
1. **Explicit phrase matching** - Fast keyword lookup (100% precision)
2. **SciBERT similarity** - Semantic matching for implicit references

Recognizes 13 categories: Grover, Shor, QFT, VQE, QAOA, Quantum Simulation, Quantum ML, Cryptography, Error Correction, Hardware, Benchmarking, Optimization, Unknown.

### OCR Enhancement

Three preprocessing strategies:
- Original image
- CLAHE contrast enhancement
- Adaptive thresholding

Selects highest confidence result per gate region.

## 📈 Dataset Statistics

| Category | Count |
|----------|-------|
| Quantum Gates Detected | 100% (250/250) |
| Problem Classified | 95.2% (238/250) |
| Descriptions Found | 98.8% (247/250) |
| Complete Metadata | 94.0% (235/250) |

## 🎯 Use Cases

- Training image-to-text models for quantum circuit captioning
- Quantum algorithm documentation generation
- Circuit diagram understanding for educational tools
- Schematic image analysis research

## ⚠️ Known Limitations

- 20% false positive rate (lattice diagrams, hardware schematics, plots)
- Struggles with hand-drawn or highly stylized circuits
- Unknown rate of false negatives (not all 8,772 images manually validated)
- Limited to 13 predefined problem categories

## 🔮 Future Improvements

- Fine-tune DINOv2 on quantum circuit domain
- Add unsupervised clustering for outlier detection
- Expand problem categories based on "Unknown" analysis
- Implement adaptive threshold adjustment for consistent quality

## 📄 License

MIT License - Academic research project

---

⭐ **Star this repo** if you're working on quantum computing or scientific image analysis!

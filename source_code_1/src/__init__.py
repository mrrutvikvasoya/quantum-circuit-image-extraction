"""
ProjectNLP: Quantum Circuit Dataset Creation Pipeline

A complete pipeline for extracting quantum circuit images from arXiv papers
with automatic gate detection and problem classification.

Modules:
    - input_handler: Read paper list from file
    - download_manager: Download PDFs from arXiv
    - pdf_image_extractor: Extract images from PDFs
    - detectors: Circuit detection, gate extraction, problem classification
    - metadata_compiler: Compile metadata for circuits
    - checkpoint_manager: Save/restore pipeline progress
    - output_generator: Generate output files
"""

__version__ = "1.0.0"
__author__ = "Rutvik Dilipbhai Vasoya"

from .utils import (
    ConfigLoader,
    PdfData,
    ImageInfo,
    DetectionResult,
    DetectionScores,
    CompleteMetadata,
    CheckpointData,
    ensureDirectory
)

__all__ = [
    'ConfigLoader',
    'PdfData',
    'ImageInfo',
    'DetectionResult',
    'DetectionScores',
    'CompleteMetadata',
    'CheckpointData',
    'ensureDirectory'
]

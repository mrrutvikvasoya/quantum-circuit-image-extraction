"""
Utility functions and classes shared across modules.
"""

import logging
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


class ConfigLoader:
    """Load and provide access to configuration settings."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, configPath: str = "config/config.yaml") -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._config is None:
            with open(configPath, 'r') as f:
                self._config = yaml.safe_load(f)
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        if self._config is None:
            self.load()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


def setupLogging(
    logFile: str = "logs/pipeline.log",
    level: str = "INFO",
    logFormat: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """Setup logging configuration."""
    
    logPath = Path(logFile)
    logPath.parent.mkdir(parents=True, exist_ok=True)
    
    numericLevel = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numericLevel,
        format=logFormat,
        handlers=[
            logging.FileHandler(logFile),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("ProjectNLP")


@dataclass
class BoundingBox:
    """Bounding box coordinates for an image."""
    x0: float
    y0: float
    x1: float
    y1: float
    
    def toDict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}
    
    @classmethod
    def fromDict(cls, data: Dict[str, float]) -> "BoundingBox":
        return cls(x0=data["x0"], y0=data["y0"], x1=data["x1"], y1=data["y1"])


@dataclass
class ImageInfo:
    """Information about an extracted image."""
    tempPath: str
    arxivId: str
    pageNumber: int
    imageIndex: int
    width: int
    height: int
    extractionMethod: str
    bbox: Optional[BoundingBox] = None
    captionText: str = ""
    captionFigureNum: Optional[int] = None
    surroundingText: str = ""  # Text within 400px of image (spatial proximity)
    referencedText: str = ""  # Text from document that references this figure (NEW)
    # New fields for Docling-based extraction
    captionPositions: Tuple[int, int] = (0, 0)  # (start, end) character positions
    surroundingPositions: Tuple[int, int] = (0, 0)  # (start, end) character positions
    referencedPositions: List[Tuple[int, int]] = field(default_factory=list)  # Character positions of referenced text
    figureNumber: Optional[int] = None  # Extracted figure number (from Docling or regex)


@dataclass
class PdfData:
    """Document-level data extracted from PDF."""
    fullText: str
    pageTexts: List[str]
    pageBoundaries: List[Tuple[int, int]]
    title: str
    abstract: str
    totalPages: int


@dataclass
class ExtractionResult:
    """Result from PDF and image extraction."""
    arxivId: str
    pdfPath: str
    success: bool
    pdfData: Optional[PdfData] = None
    images: List[ImageInfo] = field(default_factory=list)
    imageCount: int = 0
    errorMessage: Optional[str] = None


@dataclass
class DetectionScores:
    """Scores from different detection stages."""
    caption: float = 0.0
    visual: float = 0.0
    cnn: float = 0.0
    combined: float = 0.0
    # Embedding-based scores
    embedding_centroid: float = 0.0
    embedding_knn: float = 0.0
    embedding_combined: float = 0.0


@dataclass
class DetectionResult:
    """Result from quantum circuit detection."""
    isQuantumCircuit: bool
    confidence: float
    decisionTier: str
    scores: DetectionScores
    gatesFound: List[str] = field(default_factory=list)
    gateConfidenceScores: dict = field(default_factory=dict)
    problemCategory: str = 'Unknown'
    problemConfidence: str = 'LOW'
    problemScore: float = 0.0
    symbolsFound: List[str] = field(default_factory=list)
    keywordsFound: List[str] = field(default_factory=list)
    negativeFeatures: List[str] = field(default_factory=list)


@dataclass
class CompleteMetadata:
    """Complete metadata for a quantum circuit image."""
    arxivNumber: str
    pageNumber: int
    figureNumber: Optional[int]
    quantumGates: List[str]
    quantumProblem: str
    descriptions: List[str]  # List with priority: [referenced/caption/surrounding]
    textPositions: List[Tuple[int, int]]
    
    def toOutputDict(self) -> Dict[str, Any]:
        """Convert to output format with snake_case keys."""
        return {
            "arxiv_number": self.arxivNumber,
            "page_number": self.pageNumber,
            "figure_number": self.figureNumber,
            "quantum_gates": self.quantumGates,
            "quantum_problem": self.quantumProblem,
            "descriptions": self.descriptions,
            "text_positions": [list(pos) for pos in self.textPositions]
        }


@dataclass
class DownloadResult:
    """Result from paper download."""
    arxivId: str
    isQuantPh: bool
    categories: List[str]
    downloadSuccess: bool
    pdfPath: Optional[str] = None
    paperTitle: str = ""
    paperAbstract: str = ""
    authors: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class CheckpointData:
    """Checkpoint data for pipeline progress."""
    lastPaperIndex: int
    lastPaperId: str
    totalCircuitsFound: int
    papersProcessed: int
    papersSkippedNotQuantPh: int
    papersFailed: int
    timestamp: str
    pipelineVersion: str = "1.0"


def ensureDirectory(path: str) -> Path:
    """Ensure directory exists, create if not."""
    dirPath = Path(path)
    dirPath.mkdir(parents=True, exist_ok=True)
    return dirPath


def cleanArxivId(rawId: str) -> str:
    """Clean arXiv ID by removing prefix and whitespace."""
    cleanedId = rawId.strip()
    
    prefixes = ["arXiv:", "arxiv:", "ARXIV:"]
    for prefix in prefixes:
        if cleanedId.startswith(prefix):
            cleanedId = cleanedId[len(prefix):]
    
    return cleanedId.strip()


def isValidArxivId(arxivId: str) -> bool:
    """Check if string is a valid arXiv ID format."""
    newFormat = r"^\d{4}\.\d{4,5}$"
    oldFormat = r"^[a-z\-]+/\d{7}$"
    
    return bool(re.match(newFormat, arxivId) or re.match(oldFormat, arxivId))

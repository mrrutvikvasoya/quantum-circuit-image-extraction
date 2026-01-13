"""
Module: Output Generator
Generate all output files (images, JSON dataset, CSV tracking).

Output Files:
- images_<examId>/qc_001.png, qc_002.png, ... (circuit images)
- dataset_<examId>.json (metadata for all circuits)
- paper_list_counts_<examId>.csv (tracking file)

CSV Values:
- Integer: Number of circuits found in paper
- 0: Paper processed but no circuits found OR not quant-ph category
- Empty string: Paper not processed (reached 250 limit)
"""

import logging
import json
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

from utils import CompleteMetadata, ConfigLoader, ensureDirectory

logger = logging.getLogger("ProjectNLP.OutputGenerator")


class OutputGenerator:
    """
    Generate and manage all pipeline outputs.
    
    Features:
    - Save circuit images to images_<examId>/ folder
    - Collect metadata for JSON dataset
    - Track paper processing in CSV
    - Atomic writes (temp file + rename)
    """
    
    def __init__(self, examId: str = "1", outputDir: str = "output", config: ConfigLoader = None):
        """
        Initialize OutputGenerator.
        
        Args:
            examId: Exam/run identifier (default: "1")
            outputDir: Base output directory (default: "output")
            config: Optional ConfigLoader (overrides examId and outputDir if provided)
        """
        # Use config values if provided, otherwise use parameters
        if config:
            self.examId = config.get("project.examId", examId)
            outputDir = config.get("paths.output.baseDir", outputDir)
            self.imagesDir = config.get("paths.output.imagesDir", f"{outputDir}/images_{self.examId}")
            self.datasetJsonPath = config.get("paths.output.datasetJson", f"{outputDir}/dataset_{self.examId}.json")
            self.countsCsvPath = config.get("paths.output.countsCsv", f"{outputDir}/paper_list_counts_{self.examId}.csv")
        else:
            self.examId = examId
            self.imagesDir = f"{outputDir}/images_{examId}"
            self.datasetJsonPath = f"{outputDir}/dataset_{examId}.json"
            self.countsCsvPath = f"{outputDir}/paper_list_counts_{examId}.csv"
        
        # Create directories if not exist
        ensureDirectory(self.imagesDir)
        ensureDirectory(Path(self.datasetJsonPath).parent)
        
        # Internal state
        self._imageCounter: int = 0
        self._dataset: Dict[str, Dict] = {}  # filename -> metadata dict
        self._csvData: List[tuple] = []  # List of (arxiv_id, count) pairs
        self._processedPapers: set = set()  # Track which papers have CSV entries
        self._paperImageCounts: Dict[str, int] = {}  # Track images per paper for naming
        
        logger.info(f"OutputGenerator initialized: examId={self.examId}")
        logger.info(f"  Images: {self.imagesDir}")
        logger.info(f"  Dataset: {self.datasetJsonPath}")
        logger.info(f"  CSV: {self.countsCsvPath}")
    
    def saveCircuit(self, imagePath: str, metadata: CompleteMetadata) -> str:
        """
        Save a quantum circuit image and its metadata.
        
        Copies image to images_<examId>/ folder with qc_XXX.png naming.
        Adds metadata to dataset for later JSON save.
        
        Args:
            imagePath: Path to source image (temp location)
            metadata: Complete metadata for the image
            
        Returns:
            Output filename (e.g., "qc_001.png"), empty string if failed
        """
        self._imageCounter += 1
        
        # Generate filename using arxiv_id: 2401.12345_1.png, 2401.12345_2.png, ...
        arxiv_id = metadata.arxivNumber.replace("/", "_")  # Handle old-format IDs
        paper_count = self._paperImageCounts.get(arxiv_id, 0) + 1
        self._paperImageCounts[arxiv_id] = paper_count
        filename = f"{arxiv_id}_{paper_count}.png"
        outputPath = Path(self.imagesDir) / filename
        
        # Copy image to output directory
        try:
            sourcePath = Path(imagePath)
            if not sourcePath.exists():
                logger.error(f"Source image not found: {imagePath}")
                self._imageCounter -= 1  # Rollback counter
                return ""
            
            shutil.copy2(imagePath, outputPath)
            logger.debug(f"Saved circuit image: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to copy image {imagePath} to {outputPath}: {e}")
            self._imageCounter -= 1  # Rollback counter
            return ""
        
        # Add metadata to dataset (using snake_case format)
        self._dataset[filename] = metadata.toOutputDict()
        
        logger.info(f"Saved circuit: {filename} (total: {self._imageCounter})")
        return filename
    
    def updateCsv(self, arxivId: str, count: Union[int, str]) -> None:
        """
        Update CSV tracking data for a paper.
        
        CSV Values:
        - Integer: Number of circuits found in paper
        - 0: Paper processed but no circuits found OR not quant-ph category
        - Empty string ("" or None): Paper not processed (reached 250 limit)
        
        Args:
            arxivId: arXiv paper ID
            count: Circuit count (int), 0 for no circuits, or "" for not processed
        """
        if arxivId in self._processedPapers:
            # Update existing entry
            for i, (aid, _) in enumerate(self._csvData):
                if aid == arxivId:
                    self._csvData[i] = (arxivId, count)
                    logger.debug(f"CSV updated: {arxivId} = {count}")
                    break
        else:
            # Add new entry
            self._csvData.append((arxivId, count))
            self._processedPapers.add(arxivId)
            logger.debug(f"CSV added: {arxivId} = {count}")
    
    def markRemainingBlank(self, remainingIds: List[str]) -> None:
        """
        Mark remaining papers as not processed (blank/empty in CSV).
        
        Called when pipeline stops before processing all papers (reached 250 limit).
        
        Args:
            remainingIds: List of paper IDs not yet processed
        """
        for arxivId in remainingIds:
            if arxivId not in self._processedPapers:
                # Empty string indicates paper was not processed
                self._csvData.append((arxivId, ""))
                self._processedPapers.add(arxivId)
        
        logger.info(f"Marked {len(remainingIds)} papers as not processed (blank)")
    
    def saveAll(self) -> bool:
        """
        Save all outputs (JSON dataset and CSV tracking file).
        
        Uses atomic writes:
        1. Write to temp file (*.tmp)
        2. Rename to final file
        This prevents corruption if crash occurs during write.
        
        Returns:
            True if all saves successful, False otherwise
        """
        jsonSuccess = self._saveJson()
        csvSuccess = self._saveCsv()
        
        if jsonSuccess and csvSuccess:
            logger.info(f"All outputs saved successfully")
            logger.info(f"  Dataset: {len(self._dataset)} circuits")
            logger.info(f"  CSV: {len(self._csvData)} papers")
        
        return jsonSuccess and csvSuccess
    
    def _saveJson(self) -> bool:
        """
        Save JSON dataset with atomic write.
        
        Format:
        {
            "qc_001.png": {
                "arxiv_number": "...",
                "page_number": 5,
                ...
            },
            ...
        }
        """
        try:
            # Ensure directory exists
            ensureDirectory(Path(self.datasetJsonPath).parent)
            
            # Atomic write: write to temp file first
            tempPath = f"{self.datasetJsonPath}.tmp"
            
            with open(tempPath, 'w', encoding='utf-8') as f:
                json.dump(self._dataset, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            shutil.move(tempPath, self.datasetJsonPath)
            
            logger.info(f"Saved JSON dataset: {self.datasetJsonPath} ({len(self._dataset)} entries)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            # Clean up temp file
            try:
                Path(f"{self.datasetJsonPath}.tmp").unlink(missing_ok=True)
            except:
                pass
            return False
    
    def _saveCsv(self) -> bool:
        """
        Save CSV tracking file with atomic write.
        
        Format:
        arxiv_id,circuit_count
        2401.13048,3
        2507.03587,0
        2507.16669,
        
        Values:
        - Integer: circuits found
        - 0: no circuits or not quant-ph
        - Empty: not processed
        """
        try:
            # Ensure directory exists
            ensureDirectory(Path(self.countsCsvPath).parent)
            
            # Atomic write: write to temp file first
            tempPath = f"{self.countsCsvPath}.tmp"
            
            with open(tempPath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Header row
                writer.writerow(["arxiv_id", "circuit_count"])
                
                # Data rows
                for arxivId, count in self._csvData:
                    # Convert count to string representation
                    if count is None or count == "":
                        countStr = ""  # Empty string for not processed
                    else:
                        countStr = str(count)  # Integer as string
                    writer.writerow([arxivId, countStr])
            
            # Atomic rename
            shutil.move(tempPath, self.countsCsvPath)
            
            logger.info(f"Saved CSV: {self.countsCsvPath} ({len(self._csvData)} entries)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            # Clean up temp file
            try:
                Path(f"{self.countsCsvPath}.tmp").unlink(missing_ok=True)
            except:
                pass
            return False
    
    def getCurrentCount(self) -> int:
        """
        Get current circuit image count.
        
        Returns:
            Number of circuit images saved so far
        """
        return self._imageCounter
    
    def loadExisting(self) -> bool:
        """
        Load existing outputs for resume capability.
        
        Loads existing JSON dataset and CSV data if files exist.
        
        Returns:
            True if loaded successfully (or files don't exist)
        """
        # Load existing JSON dataset
        if Path(self.datasetJsonPath).exists():
            try:
                with open(self.datasetJsonPath, 'r', encoding='utf-8') as f:
                    self._dataset = json.load(f)
                logger.info(f"Loaded existing JSON: {len(self._dataset)} entries")
            except Exception as e:
                logger.warning(f"Failed to load existing JSON: {e}")
                self._dataset = {}
        
        # Load existing CSV data
        if Path(self.countsCsvPath).exists():
            try:
                with open(self.countsCsvPath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        arxivId = row["arxiv_id"]
                        countStr = row["circuit_count"]
                        
                        if countStr == "":
                            count = ""
                        else:
                            count = int(countStr)
                        
                        self._csvData.append((arxivId, count))
                        self._processedPapers.add(arxivId)
                        
                logger.info(f"Loaded existing CSV: {len(self._csvData)} entries")
            except Exception as e:
                logger.warning(f"Failed to load existing CSV: {e}")
                self._csvData = []
                self._processedPapers = set()
        
        return True
    
    def setImageCounter(self, count: int) -> None:
        """
        Set the image counter for resume.
        
        Args:
            count: Current circuit count to resume from
        """
        self._imageCounter = count
        logger.debug(f"Image counter set to: {count}")
    
    def getDatasetSize(self) -> int:
        """
        Get number of entries in dataset.
        
        Returns:
            Number of circuit entries in JSON dataset
        """
        return len(self._dataset)

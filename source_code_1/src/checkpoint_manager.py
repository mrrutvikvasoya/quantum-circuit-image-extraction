"""
Module 6: Checkpoint Manager
Save and restore pipeline progress for crash recovery and resume capability.

Checkpoint File: checkpoints/progress.json

Saved Fields:
- lastPaperIndex: int - Index of last processed paper
- lastPaperId: str - ArXiv ID of last processed paper
- totalCircuitsFound: int - Total quantum circuits found so far
- papersProcessed: int - Total papers processed
- papersSkippedNotQuantPh: int - Papers skipped (not quant-ph category)
- papersFailed: int - Papers that failed to process
- timestamp: str - ISO format datetime of checkpoint
- pipelineVersion: str - Pipeline version (default "1.0")

Atomic Writes:
- All saves write to temp file first (progress.json.tmp)
- Then rename to final file (progress.json)
- This prevents corruption if crash occurs during write
"""

import logging
import json
import shutil
from pathlib import Path
from typing import Tuple, Optional

from utils import CheckpointData, ConfigLoader, ensureDirectory

logger = logging.getLogger("ProjectNLP.CheckpointManager")


class CheckpointManager:
    """
    Manage pipeline checkpoints for crash recovery and resume capability.
    
    Features:
    - Atomic writes to prevent file corruption
    - Automatic directory creation
    - Resume point calculation (lastPaperIndex + 1)
    - Handles corrupted checkpoint files gracefully
    """
    
    def __init__(self, config: ConfigLoader = None, checkpointPath: str = "checkpoints/progress.json"):
        """
        Initialize CheckpointManager.
        
        Args:
            config: ConfigLoader instance (optional, if provided uses config path)
            checkpointPath: Path to checkpoint file (default: "checkpoints/progress.json")
        """
        if config:
            self.checkpointPath = config.get("paths.checkpoints.file", checkpointPath)
        else:
            self.checkpointPath = checkpointPath
        
        # Create directory if not exists
        ensureDirectory(Path(self.checkpointPath).parent)
        
        logger.debug(f"CheckpointManager initialized: {self.checkpointPath}")
    
    def save(self, data: CheckpointData) -> bool:
        """
        Save checkpoint data atomically.
        
        Uses atomic write pattern:
        1. Write to temp file (progress.json.tmp)
        2. Rename temp file to final file (progress.json)
        This prevents corruption if crash occurs during write.
        
        Args:
            data: CheckpointData to save
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Prepare data dictionary
            checkpointDict = {
                "lastPaperIndex": data.lastPaperIndex,
                "lastPaperId": data.lastPaperId,
                "totalCircuitsFound": data.totalCircuitsFound,
                "papersProcessed": data.papersProcessed,
                "papersSkippedNotQuantPh": data.papersSkippedNotQuantPh,
                "papersFailed": data.papersFailed,
                "timestamp": data.timestamp,
                "pipelineVersion": data.pipelineVersion
            }
            
            # Atomic write: write to temp file first
            tempPath = f"{self.checkpointPath}.tmp"
            
            with open(tempPath, 'w', encoding='utf-8') as f:
                json.dump(checkpointDict, f, indent=2)
            
            # Atomic rename: replace final file
            shutil.move(tempPath, self.checkpointPath)
            
            logger.debug(
                f"Checkpoint saved: {data.totalCircuitsFound} circuits "
                f"at paper index {data.lastPaperIndex} ({data.lastPaperId})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            try:
                tempPath = Path(f"{self.checkpointPath}.tmp")
                if tempPath.exists():
                    tempPath.unlink()
            except:
                pass
            return False
    
    def load(self) -> Optional[CheckpointData]:
        """
        Load checkpoint data from file.
        
        Behavior:
        - Returns None if file does not exist
        - Returns None if file is corrupted (invalid JSON)
        - Logs warning if file is corrupted
        
        Returns:
            CheckpointData or None if no valid checkpoint exists
        """
        checkpointFile = Path(self.checkpointPath)
        
        # Return None if file not exists
        if not checkpointFile.exists():
            logger.info("No checkpoint file found, starting fresh")
            return None
        
        try:
            with open(self.checkpointPath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            requiredFields = [
                "lastPaperIndex", "lastPaperId", "totalCircuitsFound",
                "papersProcessed", "papersSkippedNotQuantPh", "papersFailed",
                "timestamp"
            ]
            
            for field in requiredFields:
                if field not in data:
                    logger.warning(f"Checkpoint file missing field: {field}")
                    return None
            
            # Create CheckpointData object
            checkpoint = CheckpointData(
                lastPaperIndex=int(data["lastPaperIndex"]),
                lastPaperId=str(data["lastPaperId"]),
                totalCircuitsFound=int(data["totalCircuitsFound"]),
                papersProcessed=int(data["papersProcessed"]),
                papersSkippedNotQuantPh=int(data["papersSkippedNotQuantPh"]),
                papersFailed=int(data["papersFailed"]),
                timestamp=str(data["timestamp"]),
                pipelineVersion=str(data.get("pipelineVersion", "1.0"))
            )
            
            logger.info(
                f"Checkpoint loaded: {checkpoint.totalCircuitsFound} circuits, "
                f"last paper index {checkpoint.lastPaperIndex} ({checkpoint.lastPaperId})"
            )
            return checkpoint
            
        except json.JSONDecodeError as e:
            logger.warning(f"Checkpoint file is corrupted (invalid JSON): {e}")
            return None
        except KeyError as e:
            logger.warning(f"Checkpoint file missing required field: {e}")
            return None
        except ValueError as e:
            logger.warning(f"Checkpoint file has invalid data type: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def exists(self) -> bool:
        """
        Check if checkpoint file exists.
        
        Returns:
            True if checkpoint file exists, False otherwise
        """
        return Path(self.checkpointPath).exists()
    
    def clear(self) -> bool:
        """
        Clear (delete) checkpoint file.
        
        Should be called after successful pipeline completion.
        
        Returns:
            True if cleared successfully or file didn't exist,
            False if deletion failed
        """
        try:
            checkpointFile = Path(self.checkpointPath)
            if checkpointFile.exists():
                checkpointFile.unlink()
                logger.info("Checkpoint cleared (pipeline completed)")
            return True
        except Exception as e:
            logger.error(f"Failed to clear checkpoint: {e}")
            return False
    
    def getResumePoint(self) -> Tuple[int, int]:
        """
        Get the resume point from checkpoint.
        
        Logic:
        - If checkpoint exists: return (lastPaperIndex + 1, totalCircuitsFound)
        - If no checkpoint: return (0, 0)
        
        Returns:
            Tuple of (startIndex, currentCircuitCount)
            - startIndex: Index to start processing from
            - currentCircuitCount: Number of circuits already found
        """
        checkpoint = self.load()
        
        if checkpoint is None:
            # No checkpoint: start from beginning
            return (0, 0)
        
        # Resume from next paper after last checkpoint
        startIndex = checkpoint.lastPaperIndex + 1
        circuitCount = checkpoint.totalCircuitsFound
        
        logger.info(f"Resume point: paper index {startIndex}, circuits found {circuitCount}")
        
        return (startIndex, circuitCount)

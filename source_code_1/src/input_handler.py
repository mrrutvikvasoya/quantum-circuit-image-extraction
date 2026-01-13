"""
Module 1: Input Handler
Read paper list file and return ordered list of arXiv IDs.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

from utils import cleanArxivId

logger = logging.getLogger("ProjectNLP.InputHandler")


@dataclass
class PaperListResult:
    """Result from loading paper list file."""
    paperIds: List[str] = field(default_factory=list)
    totalCount: int = 0
    loadSuccess: bool = False
    errors: List[str] = field(default_factory=list)


class InputHandler:
    """Handle reading and parsing of paper list file."""
    
    def __init__(self, filePath: str):
        """
        Initialize InputHandler with file path.
        
        Args:
            filePath: Path to the paper list file
        """
        self.filePath = filePath
        self._paperIds: List[str] = []
        self._loaded = False
    
    def load(self) -> PaperListResult:
        """
        Load and parse paper IDs from file.
        
        Processes the paper list file line by line:
        - Skips empty lines and comment lines (starting with #)
        - Cleans arXiv IDs (removes prefixes, trims whitespace)
        - Validates arXiv ID format
        - Removes duplicates (keeps first occurrence)
        - Logs warnings for invalid IDs and duplicates
        
        Returns:
            PaperListResult with paper IDs and status
        """
        result = PaperListResult()
        
        # Log start of loading process
        logger.info(f"Starting to load paper list from: {self.filePath}")
        
        # Check if file exists
        path = Path(self.filePath)
        if not path.exists():
            errorMsg = f"Paper list file not found: {self.filePath}"
            logger.error(errorMsg)
            result.errors.append(errorMsg)
            return result
        
        try:
            # Read all lines from file
            with open(self.filePath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Track seen IDs for duplicate detection
            seenIds = set()
            validIds = []
            invalidCount = 0
            duplicateCount = 0
            commentCount = 0
            emptyCount = 0
            
            # Process each line
            for lineNum, line in enumerate(lines, start=1):
                # Strip whitespace
                strippedLine = line.strip()
                
                # Skip empty lines
                if not strippedLine:
                    emptyCount += 1
                    continue
                
                # Skip comment lines (starting with #)
                if strippedLine.startswith('#'):
                    commentCount += 1
                    continue
                
                # Clean the arXiv ID (remove prefixes, trim whitespace)
                cleanedId = cleanArxivId(strippedLine)
                
                # Skip if cleaning resulted in empty string
                if not cleanedId:
                    logger.warning(f"Invalid arXiv ID format at line {lineNum}: '{strippedLine}'")
                    invalidCount += 1
                    continue
                
                # Validate arXiv ID format
                from utils import isValidArxivId
                if not isValidArxivId(cleanedId):
                    logger.warning(f"Invalid arXiv ID format at line {lineNum}: '{cleanedId}'")
                    invalidCount += 1
                    continue
                
                # Check for duplicates (keep first occurrence)
                if cleanedId in seenIds:
                    logger.warning(f"Duplicate arXiv ID found at line {lineNum} (skipping): {cleanedId}")
                    duplicateCount += 1
                    continue
                
                # Add to valid IDs list and mark as seen
                seenIds.add(cleanedId)
                validIds.append(cleanedId)
            
            # Store results
            self._paperIds = validIds
            self._loaded = True
            
            result.paperIds = validIds
            result.totalCount = len(validIds)
            result.loadSuccess = True
            
            # Log completion with statistics
            logger.info(f"Successfully loaded paper list: {self.filePath}")
            logger.info(f"Total papers in list: {result.totalCount}")
            logger.info(
                f"Processing statistics: "
                f"{len(validIds)} valid IDs, "
                f"{invalidCount} invalid, "
                f"{duplicateCount} duplicates, "
                f"{commentCount} comments, "
                f"{emptyCount} empty lines"
            )
            
            # Handle edge cases
            if result.totalCount == 0:
                if len(lines) == 0:
                    logger.warning("Paper list file is empty")
                elif commentCount > 0 and invalidCount == 0 and duplicateCount == 0:
                    logger.warning("Paper list contains only comments, no valid IDs found")
                elif invalidCount > 0:
                    logger.warning(f"No valid arXiv IDs found (all {invalidCount} IDs were invalid)")
                else:
                    logger.warning("No valid arXiv IDs found in paper list")
            
        except Exception as e:
            errorMsg = f"Error reading paper list file: {str(e)}"
            logger.error(errorMsg)
            result.errors.append(errorMsg)
        
        return result
    
    def getPaperIds(self) -> List[str]:
        """Get list of paper IDs."""
        if not self._loaded:
            self.load()
        return self._paperIds
    
    def getPaperCount(self) -> int:
        """Get total count of papers."""
        if not self._loaded:
            self.load()
        return len(self._paperIds)

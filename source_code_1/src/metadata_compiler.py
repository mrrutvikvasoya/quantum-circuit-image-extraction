"""
Module 5: Metadata Compiler
Compile all required metadata fields for quantum circuit images.

Output JSON Format:
{
    "qc_001.png": {
        "arxiv_number": "2401.13048",     # ArXiv paper ID (string)
        "page_number": 5,                  # Page where image found (integer)
        "figure_number": 3,                # Figure number in paper (integer)
        "quantum_gates": ["H", "CNOT"],    # List of gates in image (list of strings)
        "quantum_problem": "VQE",          # Problem/algorithm type (string)
        "descriptions": ["Figure 3.."],    # Descriptive texts (list of strings)
        "text_positions": [[0, 50], ...]   # Text positions as [start, end] (list of tuples)
    }
}

Text Positions Meaning:
- Each tuple (beginning, end) represents character offsets in the concatenated PDF text
- beginning: 0-indexed starting character position of the description text
- end: character position immediately after the last character
- Used to locate the exact source of each description in the original document
"""

import logging
import re
from typing import Dict, List, Tuple, Optional

from utils import (
    CompleteMetadata, ImageInfo, DetectionResult, PdfData, ConfigLoader
)

logger = logging.getLogger("ProjectNLP.MetadataCompiler")


class MetadataCompiler:
    """
    Compile metadata for quantum circuit images.
    
    Integrates data from:
    - ImageInfo: Source image details (page, caption, arXiv ID)
    - DetectionResult: Gates extracted by OCR + problem classification from SciBERT
    - PdfData: Full document text for position extraction
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize MetadataCompiler.
        
        Args:
            config: ConfigLoader instance
        """
        self.algorithmMapping = config.get("quantumAlgorithms", {})
        self.gateNormalization = config.get("gateNormalization", {})
    
    def compile(
        self,
        imageInfo: ImageInfo,
        detectionResult: DetectionResult,
        pdfData: Optional[PdfData]
    ) -> CompleteMetadata:
        """
        Compile complete metadata for a quantum circuit image.
        
        Args:
            imageInfo: Information about the extracted image (with Docling data)
            detectionResult: Result from detection pipeline (includes gates and problem)
            pdfData: Document-level PDF data
            
        Returns:
            CompleteMetadata with all required fields
        """
        # Use figure number from Docling/spatial proximity extraction
        figureNumber = imageInfo.figureNumber
        
        # Get gates from detection result (OCR-extracted)
        quantumGates = self._normalizeGates(detectionResult.gatesFound)
        
        # Optionally supplement from text analysis
        if pdfData and len(quantumGates) < 2:
            textGates = self._extractGatesFromText(pdfData, imageInfo.pageNumber)
            quantumGates = list(set(quantumGates + textGates))
        
        # Get problem classification from detection result (SciBERT-classified)
        if detectionResult.problemCategory and detectionResult.problemCategory != 'Unknown':
            quantumProblem = detectionResult.problemCategory
        else:
            # Fallback to text-based identification
            quantumProblem = self._identifyQuantumProblem(
                imageInfo.captionText,
                imageInfo.surroundingText,
                detectionResult.keywordsFound
            )
        
        
        # =====================================================================
        # Description Selection (SINGLE SOURCE - NO CONCATENATION)
        # =====================================================================
        # Pick exactly ONE description source to maintain position alignment.
        # The text must be copied exactly from the PDF - no modifications.
        # Priority: 1) Figure references  2) Caption  3) Surrounding text
        # =====================================================================
        
        description = ""
        textPositions = []
        
        # Priority 1: Figure references from document (most context, explains the figure)
        if hasattr(imageInfo, 'referencedText') and imageInfo.referencedText:
            ref_text = imageInfo.referencedText.strip()
            # Reduced minimum: 15 chars (was 20)
            if len(ref_text) >= 15:
                description = ref_text
                textPositions = imageInfo.referencedPositions if imageInfo.referencedPositions else []
                logger.debug(f"Using referencedText as description ({len(description)} chars)")
        
        # Priority 2: Caption text (if no valid referenced text)
        if not description and imageInfo.captionText:
            caption_text = imageInfo.captionText.strip()
            # Reduced minimum: 5 chars (was 10) - even short captions are useful
            if len(caption_text) >= 5:
                description = caption_text
                # captionPositions is a tuple (start, end), wrap in list
                if imageInfo.captionPositions and imageInfo.captionPositions != (0, 0):
                    textPositions = [imageInfo.captionPositions]
                else:
                    textPositions = []
                logger.debug(f"Using captionText as description ({len(description)} chars)")
        
        # Priority 3: Surrounding text (ALWAYS use if available, even if very short)
        # Critical for equation-based circuits without captions
        if not description and imageInfo.surroundingText:
            surrounding_text = imageInfo.surroundingText.strip()
            # Accept ANY non-empty text (no minimum)
            if len(surrounding_text) > 0:
                description = surrounding_text
                # surroundingPositions is a tuple (start, end), wrap in list
                if imageInfo.surroundingPositions and imageInfo.surroundingPositions != (0, 0):
                    textPositions = [imageInfo.surroundingPositions]
                else:
                    textPositions = []
                logger.debug(f"Using surroundingText as description ({len(description)} chars)")
        
        # Priority 4: Fallback - search document for figure references
        if not description and pdfData and figureNumber:
            figRefs = self._findFigureReferences(pdfData.fullText, figureNumber)
            if figRefs:
                # Pick the first (and typically best) reference
                ref_text, ref_pos = figRefs[0]
                if len(ref_text) >= 20:
                    description = ref_text
                    textPositions = [ref_pos]
                    logger.debug(f"Using fallback figure reference as description ({len(description)} chars)")
        
        logger.debug(f"Compiled metadata: gates={quantumGates}, problem={quantumProblem}")
        logger.debug(f"Final description: {len(description)} chars, positions: {textPositions}")
        
        # Return metadata with single description (no concatenation)
        return CompleteMetadata(
            arxivNumber=imageInfo.arxivId,
            pageNumber=imageInfo.pageNumber,
            figureNumber=figureNumber,
            quantumGates=quantumGates,
            quantumProblem=quantumProblem,
            descriptions=[description] if description else [],
            textPositions=textPositions if description else []
        )
    
    def _extractFigureNumber(self, captionText: str) -> Optional[int]:
        """
        Extract figure number from caption text.
        
        Patterns matched:
        - "Figure 3", "Fig. 3", "FIG. 3"
        """
        if not captionText:
            return None
        
        patterns = [
            r'Figure\s+(\d+)',
            r'Fig\.\s*(\d+)',
            r'FIG\.\s*(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, captionText, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _normalizeGates(self, gates: List[str]) -> List[str]:
        """
        Normalize gate names and remove duplicates.
        
        Standard gate names:
        - Single qubit: H, X, Y, Z, S, T, I
        - Rotation: RX, RY, RZ
        - Two qubit: CNOT, CX, CY, CZ, SWAP
        - Three qubit: CCX, CSWAP
        - Measurement: M, MEASURE
        """
        normalized = []
        seen = set()
        
        for gate in gates:
            gateUpper = gate.upper().strip()
            
            # Apply any custom normalization
            gateLower = gate.lower()
            if gateLower in self.gateNormalization:
                gateUpper = self.gateNormalization[gateLower]
            
            # Standard single-letter gates stay uppercase
            if gateUpper in {'H', 'X', 'Y', 'Z', 'S', 'T', 'I', 'M'}:
                pass  # Already uppercase
            
            # Normalize common variations
            if gateUpper in {'CX', 'CONTROLLED-X', 'CONTROLLED-NOT'}:
                gateUpper = 'CNOT'
            elif gateUpper in {'CCX', 'TOFFOLI'}:
                gateUpper = 'CCX'
            elif gateUpper in {'MEASURE', 'MEASUREMENT'}:
                gateUpper = 'M'
            
            if gateUpper not in seen:
                seen.add(gateUpper)
                normalized.append(gateUpper)
        
        return normalized
    
    def _extractGatesFromText(self, pdfData: PdfData, pageNumber: int) -> List[str]:
        """Extract gate names from surrounding text as supplement."""
        gates = []
        
        if not pdfData or pageNumber > len(pdfData.pageTexts):
            return gates
        
        pageText = pdfData.pageTexts[pageNumber - 1] if pageNumber > 0 else ""
        
        gatePatterns = [
            (r'\b(Hadamard)\s+gate', 'H'),
            (r'\b(CNOT|CX|controlled-NOT)\b', 'CNOT'),
            (r'\b(Toffoli|CCX)\b', 'CCX'),
            (r'\b(SWAP)\s+gate', 'SWAP'),
            (r'\b(Pauli-X|Pauli X)\b', 'X'),
            (r'\b(Pauli-Y|Pauli Y)\b', 'Y'),
            (r'\b(Pauli-Z|Pauli Z)\b', 'Z'),
        ]
        
        for pattern, gateName in gatePatterns:
            if re.search(pattern, pageText, re.IGNORECASE):
                if gateName not in gates:
                    gates.append(gateName)
        
        return gates
    
    def _identifyQuantumProblem(
        self,
        captionText: str,
        surroundingText: str,
        keywordsFound: List[str]
    ) -> str:
        """
        Identify the quantum problem or algorithm from text.
        Fallback method when SciBERT classification not available.
        """
        combinedText = f"{captionText} {surroundingText}".lower()
        
        # Check algorithm mapping from config
        for keyword, algorithmName in self.algorithmMapping.items():
            if keyword.lower() in combinedText:
                return algorithmName
        
        # Check keywords from detection
        for keyword in keywordsFound:
            keyLower = keyword.lower()
            if keyLower in self.algorithmMapping:
                return self.algorithmMapping[keyLower]
        
        # Hardcoded fallback patterns
        fallbackPatterns = {
            'vqe': 'VQE',
            'variational quantum eigensolver': 'VQE',
            'qaoa': 'QAOA',
            'quantum approximate optimization': 'QAOA',
            'grover': 'Grover_Algorithm',
            'quantum search': 'Grover_Algorithm',
            'shor': 'Shor_Algorithm',
            'factorization': 'Shor_Algorithm',
            'qft': 'QFT_QPE',
            'quantum fourier': 'QFT_QPE',
            'phase estimation': 'QFT_QPE',
        }
        
        for pattern, problem in fallbackPatterns.items():
            if pattern in combinedText:
                return problem
        
        return "Unknown"
    
    def _extractDescriptions(
        self,
        imageInfo: ImageInfo,
        pdfData: Optional[PdfData],
        figureNumber: Optional[int]
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Extract descriptive text parts and their positions in the document.
        
        Text Positions Meaning:
        - Each tuple (beginning, end) represents character offsets
        - beginning: 0-indexed starting position in the concatenated full PDF text
        - end: position immediately after the last character
        - Example: text_positions = [(0, 100), (500, 650)]
          means the first description spans characters 0-99,
          and the second spans characters 500-649 in fullText
        
        Returns:
            Tuple of (descriptions list, text positions list)
        """
        descriptions = []
        textPositions = []
        
        # Add caption text as first description
        if imageInfo.captionText:
            cleanCaption = imageInfo.captionText.strip()
            descriptions.append(cleanCaption)
            
            if pdfData:
                position = self._findTextPosition(
                    pdfData.fullText,
                    cleanCaption
                )
                if position:
                    textPositions.append(position)
                else:
                    # Placeholder position if not found
                    textPositions.append((0, 0))
            else:
                textPositions.append((0, 0))
        
        # Add surrounding text if available
        if imageInfo.surroundingText:
            cleanSurrounding = imageInfo.surroundingText.strip()
            if cleanSurrounding and cleanSurrounding not in descriptions:
                descriptions.append(cleanSurrounding)
                
                if pdfData:
                    position = self._findTextPosition(
                        pdfData.fullText,
                        cleanSurrounding
                    )
                    if position:
                        textPositions.append(position)
                    else:
                        textPositions.append((0, 0))
                else:
                    textPositions.append((0, 0))
        
        # Find figure references in the document
        if pdfData and figureNumber:
            figRefs = self._findFigureReferences(
                pdfData.fullText,
                figureNumber
            )
            
            for refText, position in figRefs:
                if refText not in descriptions:
                    descriptions.append(refText)
                    textPositions.append(position)
        
        return descriptions, textPositions
    
    def _findTextPosition(
        self,
        fullText: str,
        searchText: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find position of text within full document text.
        
        IMPORTANT: Returns positions relative to ORIGINAL text, not normalized text.
        
        Args:
            fullText: Original full document text
            searchText: Text to search for
        
        Returns:
            Tuple (start_position, end_position) in ORIGINAL text, or None if not found
        """
        if not searchText or not fullText:
            return None
        
        # Try 1: Exact match in original text (fastest)
        startPos = fullText.find(searchText)
        if startPos >= 0:
            endPos = startPos + len(searchText)
            return (startPos, endPos)
        
        # Try 2: Normalized search text in original text
        cleanSearch = ' '.join(searchText.split())
        startPos = fullText.find(cleanSearch)
        if startPos >= 0:
            endPos = startPos + len(cleanSearch)
            return (startPos, endPos)
        
        # Try 3: Flexible whitespace matching with regex
        # This handles cases where original text has extra spaces/newlines
        try:
            # Escape special regex characters, but allow flexible whitespace
            pattern = re.escape(cleanSearch).replace(r'\ ', r'\s+')
            match = re.search(pattern, fullText, re.IGNORECASE)
            if match:
                return (match.start(), match.end())
        except Exception:
            pass
        
        # Try 4: Partial match with first 50 characters (fallback)
        if len(cleanSearch) > 50:
            firstPart = cleanSearch[:50]
            try:
                pattern = re.escape(firstPart).replace(r'\ ', r'\s+')
                match = re.search(pattern, fullText, re.IGNORECASE)
                if match:
                    # Return approximate end position
                    approxEnd = match.start() + len(cleanSearch)
                    return (match.start(), min(approxEnd, len(fullText)))
            except Exception:
                pass
        
        return None
    
    def _findFigureReferences(
        self,
        fullText: str,
        figureNumber: int
    ) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Find sentences that reference a specific figure.
        
        Returns:
            List of (reference_text, (start_pos, end_pos))
        """
        references = []
        
        patterns = [
            rf'[^.]*Figure\s+{figureNumber}\b[^.]*\.',
            rf'[^.]*Fig\.\s*{figureNumber}\b[^.]*\.',
            rf'[^.]*shown in Fig\.\s*{figureNumber}[^.]*\.',
            rf'[^.]*illustrated in Figure\s+{figureNumber}[^.]*\.',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, fullText, re.IGNORECASE)
            for match in matches:
                refText = match.group(0).strip()
                startPos = match.start()
                endPos = match.end()
                
                # Skip very long matches (likely parsing error)
                if len(refText) > 500:
                    continue
                
                # Skip if it's just the caption itself
                if re.match(rf'^Figure\s+{figureNumber}\s*[:.]?\s*$', refText, re.IGNORECASE):
                    continue
                
                references.append((refText, (startPos, endPos)))
        
        # Return at most 5 references
        return references[:5]

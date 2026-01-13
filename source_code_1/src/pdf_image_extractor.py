"""
Comprehensive PDF Image Extractor using PyMuPDF
Implements 3-method extraction: Embedded + Vector + Rendered with filtering
PLUS comprehensive metadata extraction with spatial text analysis
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
import logging
import re
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from utils import (
    ConfigLoader, ImageInfo, PdfData, ExtractionResult, BoundingBox
)
from detectors.matrix_detector import MatrixDetector

logger = logging.getLogger("ProjectNLP.PdfImageExtractor")


class PdfImageExtractor:
    """
    Comprehensive PDF image extractor with metadata extraction.
    
    Extracts:
    1. Images (3 methods: embedded, vector, rendered)
    2. Captions (spatial proximity search)
    3. Figure numbers (regex with roman numerals support)
    4. Surrounding text (radius-based search)
    """
    
    def __init__(self, config: ConfigLoader):
        """Initialize extractor with configuration."""
        self.config = config
        self.dpi = 300  # Rendering DPI
        self.merge_threshold = 50  # Pixels for grouping drawings
        
        # Metadata extraction parameters (optimized values)
        self.caption_search_radius = 200  # pixels below image
        self.surrounding_text_radius = 400  # pixels around image
        self.caption_min_overlap = 0.3  # 30% horizontal overlap
        
        # Matrix detector for filtering
        self.matrix_detector = MatrixDetector()
        
        logger.info("PdfImageExtractor initialized with 3-method approach + metadata extraction")
    
    def extract(
        self,
        pdfPath: str,
        arxivId: str,
        paperTitle: str = "",
        paperAbstract: str = ""
    ) -> ExtractionResult:
        """
        Extract images and metadata using comprehensive approach.
        
        Args:
            pdfPath: Path to PDF file
            arxivId: arXiv ID
            paperTitle: Paper title from arXiv API
            paperAbstract: Paper abstract from arXiv API
            
        Returns:
            ExtractionResult with images and complete metadata
        """
        logger.info(f"Extracting from PDF: {pdfPath}")
        
        try:
            # Check for empty/corrupted PDF
            pdf_size = Path(pdfPath).stat().st_size
            if pdf_size == 0:
                logger.error(f"PDF file is empty: {pdfPath}")
                return ExtractionResult(
                    arxivId=arxivId,
                    pdfPath=pdfPath,
                    success=False,
                    images=[],
                    pdfData=None,
                    errorMessage="Empty PDF file"
                )
            
            doc = fitz.open(pdfPath)
            
            # Create temp output directory
            temp_dir = Path("temp/images") / arxivId
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            all_figures = []
            
            # =====================================================================
            # STEP 1: Extract PDF data FIRST to get page boundaries
            # This gives us document-wide character offsets for each page
            # =====================================================================
            pdf_data = self._extract_pdf_data(doc, paperTitle, paperAbstract)
            
            # =====================================================================
            # STEP 2: Extract text blocks with DOCUMENT-WIDE positions
            # Add page boundary offset so positions align with fullText
            # =====================================================================
            page_text_blocks = {}
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get the document-wide starting position for this page
                page_offset = pdf_data.pageBoundaries[page_num][0] if page_num < len(pdf_data.pageBoundaries) else 0
                
                try:
                    # Extract text blocks with per-page positions
                    page_blocks = self._extract_text_blocks_with_positions(page)
                    
                    # Convert to document-wide positions by adding page offset
                    for block in page_blocks:
                        block['char_start'] += page_offset
                        block['char_end'] += page_offset
                    
                    page_text_blocks[page_num] = page_blocks
                    
                except RuntimeError as e:
                    if "stack overflow" in str(e):
                        logger.warning(f"MuPDF stack overflow on page {page_num} of {arxivId}, skipping text extraction for this page")
                        page_text_blocks[page_num] = []  # Empty text blocks for this page
                    else:
                        raise  # Re-raise if it's a different error
            
            # Process each page with all 3 methods
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Method 1: Embedded images
                embedded = self._extract_embedded_images(
                    page, page_num, arxivId, temp_dir
                )
                all_figures.extend(embedded)
                
                # Method 2: Vector graphics
                vector = self._extract_vector_regions(
                    page, page_num, arxivId, temp_dir
                )
                all_figures.extend(vector)
                
                # Method 3: Rendered regions (with filtering)
                rendered = self._extract_rendered_figures(
                    page, page_num, arxivId, temp_dir
                )
                all_figures.extend(rendered)
            
            # Deduplicate figures
            unique_figures = self._deduplicate_figures(all_figures)
            
            # Convert to ImageInfo objects WITH METADATA
            # NOTE: pdf_data is already extracted above, no need to extract again
            images = self._create_image_info_with_metadata(
                unique_figures, arxivId, page_text_blocks, pdf_data
            )
            
            doc.close()
            
            logger.info(f"Extracted {len(images)} images with metadata from {arxivId}")
            
            return ExtractionResult(
                arxivId=arxivId,
                pdfPath=pdfPath,
                success=True,
                pdfData=pdf_data,
                images=images,
                imageCount=len(images)
            )
            
        except Exception as e:
            logger.error(f"Extraction failed for {arxivId}: {e}")
            import traceback
            traceback.print_exc()
            return ExtractionResult(
                arxivId=arxivId,
                pdfPath=pdfPath,
                success=False,
                errorMessage=str(e)
            )
    
    def _extract_text_blocks_with_positions(self, page: fitz.Page) -> List[Dict]:
        """
        Extract text blocks with precise bounding boxes and positions.
        
        Returns list of:
        {
            'text': "Figure 1: Quantum circuit...",
            'bbox': (x0, y0, x1, y1),
            'char_start': 1234,
            'char_end': 1280
        }
        """
        text_blocks = []
        char_position = 0
        
        # Get structured text with positions
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip non-text blocks
                continue
            
            # Extract text from block
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")
                block_text += " "
            
            block_text = block_text.strip()
            
            if not block_text:
                continue
            
            bbox = block.get("bbox", (0, 0, 0, 0))
            
            text_blocks.append({
                'text': block_text,
                'bbox': bbox,
                'char_start': char_position,
                'char_end': char_position + len(block_text)
            })
            
            char_position += len(block_text) + 1
        
        return text_blocks
    
    def _find_caption(
        self, image_bbox: Tuple[float, float, float, float],
        text_blocks: List[Dict]
    ) -> Optional[Dict]:
        """
        Find caption text below image using spatial proximity.
        
        Strategy:
        1. Search below image within caption_search_radius
        2. Check horizontal alignment
        3. Look for "Figure" pattern
        4. Return closest match
        """
        if not image_bbox:
            return None
        
        image_x0, image_y0, image_x1, image_y1 = image_bbox
        image_width = image_x1 - image_x0
        
        candidates = []
        
        for block in text_blocks:
            block_x0, block_y0, block_x1, block_y1 = block['bbox']
            
            # Must be below image
            if block_y0 < image_y1:
                continue
            
            # Within search radius?
            distance = block_y0 - image_y1
            if distance > self.caption_search_radius:
                continue
            
            # Check horizontal alignment (overlap)
            x_overlap = min(image_x1, block_x1) - max(image_x0, block_x0)
            if x_overlap < image_width * self.caption_min_overlap:
                continue
            
            # Contains figure pattern?
            text = block['text']
            if re.search(r'\b(Figure|Fig\.?|FIG\.?)\s+', text, re.IGNORECASE):
                # Clean the caption text
                cleaned_text = self._clean_caption_text(text)
                
                # Validate: Skip if it looks like a matrix
                if self._is_matrix_text(cleaned_text):
                    continue  # Skip matrix notation
                
                candidates.append({
                    'text': cleaned_text,  # Use cleaned text
                    'distance': distance,
                    'bbox': block['bbox'],
                    'char_start': block['char_start'],
                    'char_end': block['char_end']
                })
        
        # Return closest caption
        if candidates:
            return min(candidates, key=lambda x: x['distance'])
        
        return None
    
    def _clean_caption_text(self, raw_text: str) -> str:
        """
        Clean caption text to extract only the relevant caption.
        
        Problems to fix:
        1. Captions include entire paragraphs
        2. Mathematical equations mixed in
        3. Subfigure markers like "(a) (b) (c)"
        4. Random symbols like "= ="
        
        Strategy:
        1. Find "Figure X:" pattern
        2. Extract up to first sentence or reasonable length
        3. Remove noise
        """
        if not raw_text:
            return ""
        
        # Find the start of the caption (Figure/Fig pattern)
        match = re.search(r'\b(Figure|Fig\.?|FIG\.?)\s+', raw_text, re.IGNORECASE)
        if not match:
            return raw_text[:500]  # Fallback: first 500 chars
        
        # Start from the Figure pattern
        caption_start = match.start()
        caption_text = raw_text[caption_start:]
        
        # Strategy 1: Extract up to first double newline (paragraph break)
        if '\n\n' in caption_text:
            caption_text = caption_text.split('\n\n')[0]
        
        # Strategy 2: Extract up to first sentence after figure description
        # Look for sentence ending after at least 50 characters
        if len(caption_text) > 50:
            # Find first period followed by space and capital letter (new sentence)
            sentence_match = re.search(r'\.(\s+[A-Z])', caption_text[50:])
            if sentence_match:
                # Cut at the period
                cut_point = 50 + sentence_match.start() + 1
                caption_text = caption_text[:cut_point]
        
        # Strategy 3: Limit to reasonable length (max 500 chars)
        if len(caption_text) > 500:
            # Try to cut at last sentence within 500 chars
            truncated = caption_text[:500]
            last_period = truncated.rfind('.')
            if last_period > 100:  # At least 100 chars
                caption_text = truncated[:last_period + 1]
            else:
                caption_text = truncated + "..."
        
        # Clean up noise
        # Remove standalone subfigure markers at the end
        caption_text = re.sub(r'\s+\([a-z]\)\s*$', '', caption_text)
        caption_text = re.sub(r'\s+\(a\)\s*\(b\)\s*\(c\)\s*$', '', caption_text)
        
        # Remove mathematical noise at the end
        caption_text = re.sub(r'\s*=\s*=\s*$', '', caption_text)
        caption_text = re.sub(r'\s*=\s*\([a-z]\)\s*\([a-z]\)\s*=\s*$', '', caption_text)
        
        # Clean up whitespace
        caption_text = re.sub(r'\s+', ' ', caption_text)  # Normalize whitespace
        caption_text = caption_text.strip()
        
        return caption_text
    
    def _is_matrix_text(self, text: str) -> bool:
        """
        Check if text looks like matrix notation rather than a caption.
        
        Matrix indicators:
        1. High density of numbers and mathematical symbols
        2. Low density of regular words
        3. Repeated patterns of numbers
        4. Contains matrix notation like "1 0 0 1" or "−1 1 0"
        
        Returns:
            True if text looks like a matrix, False otherwise
        """
        if not text or len(text) < 10:
            return False
        
        # Count different character types
        total_chars = len(text.replace(' ', ''))  # Exclude spaces
        if total_chars == 0:
            return False
        
        # Count numbers (including negative signs)
        numbers = len(re.findall(r'[-−]?\d+', text))
        
        # Count mathematical symbols
        math_symbols = len(re.findall(r'[=+\-−×÷√∑∏∫]', text))
        
        # Count regular words (3+ letters)
        words = len(re.findall(r'\b[a-zA-Z]{3,}\b', text))
        
        # Matrix indicators:
        # 1. High number density (> 40% of text is numbers)
        number_chars = sum(len(m.group()) for m in re.finditer(r'[-−]?\d+', text))
        number_density = number_chars / total_chars if total_chars > 0 else 0
        
        # 2. Low word count (< 5 words)
        # 3. High math symbol count
        
        # Decision logic
        if number_density > 0.4:  # > 40% numbers
            return True
        
        if numbers > 10 and words < 5:  # Many numbers, few words
            return True
        
        if math_symbols > 5 and words < 3:  # Many symbols, very few words
            return True
        
        # Check for repeated number patterns (matrix rows)
        # e.g., "1 0 0 1" or "1 −1 0 0"
        if re.search(r'(\d+\s+){4,}', text):  # 4+ numbers in a row
            return True
        
        return False
    
    
    def _find_equation_number(
        self, image_bbox: tuple, text_blocks: List[Dict]
    ) -> tuple:
        """
        Find equation number positioned near the image.
        
        Returns:
            (number, full_text): e.g. ("15", "(15)") or (None, None)
        """
        if not image_bbox or not text_blocks:
            return (None, None)
        
        img_x0, img_y0, img_x1, img_y1 = image_bbox
        img_height = img_y1 - img_y0
        
        
        candidates = []
        
        for block in text_blocks:
            bbox = block.get('bbox')
            if not bbox:
                continue
            
            bx0, by0, bx1, by1 = bbox
            block_center_y = (by0 + by1) / 2
            text = block['text'].strip()
            
            # Try multiple equation patterns
            patterns = [
                (r'^\s*\((\d+(?:\.\d+)?)\)\s*$', 1),           # (15)
                (r'^\s*Eq\.\s*\((\d+(?:\.\d+)?)\)\s*$', 1),    # Eq. (15)
                (r'^\s*Equation\s*\((\d+(?:\.\d+)?)\)\s*$', 1), # Equation (15)
                (r'^\s*\[(\d+(?:\.\d+)?)\]\s*$', 1),           # [15]
            ]
            
            eq_num = None
            for pattern, group_idx in patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    eq_num = match.group(group_idx)
                    break
            
            if not eq_num:
                continue
            
            # Spatial: check right, left, or below
            is_to_right = bx0 >= img_x1 - 30
            h_dist_right = abs(bx0 - img_x1)
            is_to_left = bx1 <= img_x0 + 30
            h_dist_left = abs(img_x0 - bx1)
            is_below = by0 >= img_y1 - 30
            v_dist_below = abs(by0 - img_y1)
            
            is_h_close = (is_to_right and h_dist_right < 150) or (is_to_left and h_dist_left < 150)
            is_v_aligned = block_center_y >= img_y0 + img_height * 0.25
            is_v_close = is_below and v_dist_below < 100
            is_h_aligned = (bx0 >= img_x0 - 50 and bx0 <= img_x1 + 50)
            
            is_valid = (is_h_close and is_v_aligned) or (is_v_close and is_h_aligned)
            
            if is_valid:
                distance = min(h_dist_right if is_to_right else 999,
                              h_dist_left if is_to_left else 999,
                              v_dist_below if is_below else 999)
                candidates.append((eq_num, text, distance))
        
        if candidates:
            candidates.sort(key=lambda x: x[2])
            best_num, best_text, _ = candidates[0]
            logger.debug(f"  Found equation: num={best_num}, text='{best_text}'")
            return (best_num, best_text)
        
        return (None, None)
    
    
    def _extract_figure_number(self, caption_text: str) -> Optional[int]:
        """
        Extract figure number from caption as INTEGER.
        
        Supports:
        - "Figure 1", "Fig. 2", "FIG. 3" → 1, 2, 3
        - Sub-figures: "1a", "1(b)", "1-a" → 1 (ignores subfigure)
        - Roman numerals: "Figure I", "Fig. IV" → 1, 4
        
        Returns:
            Integer figure number or None
        """
        if not caption_text:
            return None
        
        # Pattern 1: Arabic numerals (extract main number only)
        patterns = [
            r'\bFigure\s+(\d+)',
            r'\bFig\.?\s+(\d+)',
            r'\bFIG\.?\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, caption_text, re.IGNORECASE)
            if match:
                return int(match.group(1))  # Return as integer
        
        # Pattern 2: Roman numerals
        roman_pattern = r'\b(?:Figure|Fig\.?|FIG\.?)\s+([IVXivx]+)'
        match = re.search(roman_pattern, caption_text, re.IGNORECASE)
        if match:
            roman = match.group(1).upper()
            # Convert to integer
            roman_map = {
                'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
                'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
            }
            return roman_map.get(roman, None)
        
        return None
    
    def _remove_figure_prefix(
        self, caption_text: str, char_start: int, char_end: int
    ) -> Tuple[str, int, int]:
        """
        Remove 'Figure X:' or 'Fig. X:' prefix from caption text.
        Updates character positions accordingly.
        
        Args:
            caption_text: Original caption text
            char_start: Original start position
            char_end: Original end position
            
        Returns:
            (cleaned_text, new_char_start, new_char_end)
        """
        if not caption_text:
            return (caption_text, char_start, char_end)
        
        # Find the figure prefix pattern
        patterns = [
            r'^\s*Figure\s+\d+[a-z]?(?:\([a-z]\))?(?:-[a-z])?[:\.\s]+',
            r'^\s*Fig\.?\s+\d+[a-z]?(?:\([a-z]\))?(?:-[a-z])?[:\.\s]+',
            r'^\s*FIG\.?\s+\d+[a-z]?(?:\([a-z]\))?(?:-[a-z])?[:\.\s]+',
            r'^\s*Figure\s+[IVXivx]+[:\.\s]+',  # Roman numerals
            r'^\s*Fig\.?\s+[IVXivx]+[:\.\s]+',
            r'^\s*FIG\.?\s+[IVXivx]+[:\.\s]+',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, caption_text, re.IGNORECASE)
            if match:
                # Remove the prefix
                prefix_length = len(match.group(0))
                cleaned_text = caption_text[prefix_length:].strip()
                
                # Update positions
                new_char_start = char_start + prefix_length
                
                # Adjust for the strip() operation
                # Count leading whitespace that was removed
                stripped_leading = len(caption_text[prefix_length:]) - len(cleaned_text)
                new_char_start += stripped_leading
                
                return (cleaned_text, new_char_start, char_end)
        
        # No prefix found, return original
        return (caption_text, char_start, char_end)
    
    
    def _find_figure_references(
        self,
        figure_number: str,
        all_text_blocks: List[Dict],
        caption_text: str = ""
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Find text from entire document that references this figure.
        
        Args:
            figure_number: Figure number (e.g., "10", "5a")
            all_text_blocks: All text blocks from entire document
            caption_text: Caption text to exclude
            
        Returns:
            (referenced_text, positions)
        """
        if not figure_number or not all_text_blocks:
            return ("", [])
        
        # Check if this is an equation reference (will be string like "Eq.15" when passed)
        # But we now pass just integers, so check if we should look for equation patterns
        is_equation_num = isinstance(figure_number, str) and 'Eq' in str(figure_number)
        
        # Convert to string for pattern matching
        fig_num_str = str(figure_number)
        
        if is_equation_num:
            # Extract just the number
            eq_num = fig_num_str.replace('Eq.', '').replace('Eq', '')
            logger.debug(f"Searching for equation {eq_num} references")
            
            # Equation reference patterns - MUST have Eq/Equation prefix
            # Don't match bare (15) or [15] - could be citations/references
            patterns = [
                rf'Eq\.\s*\(?\s*{re.escape(eq_num)}\s*\)?',      # Eq. (15) or Eq. 15
                rf'Equation\s*\(?\s*{re.escape(eq_num)}\s*\)?',  # Equation (15) or Equation 15
                rf'eq\.\s*\(?\s*{re.escape(eq_num)}\s*\)?',      # eq. (15) (lowercase)
                rf'equation\s*\(?\s*{re.escape(eq_num)}\s*\)?',  # equation 15 (lowercase)
            ]
        else:
            # Regular figure reference patterns
            logger.debug(f"Searching for Figure {fig_num_str} references")
            
            patterns = [
                # With period and space
                rf'\bFig\.\s*{re.escape(fig_num_str)}\b',
                rf'\bFigure\s+{re.escape(fig_num_str)}\b',
                # Without period (Fig 1)
                rf'\bFig\s+{re.escape(fig_num_str)}\b',
                # No space (Fig.1, Figure1)
                rf'\bFig\.{re.escape(fig_num_str)}\b',
                rf'\bFigure{re.escape(fig_num_str)}\b',
                # Lowercase variants
                rf'\bfig\.\s*{re.escape(fig_num_str)}\b',
                rf'\bfigure\s+{re.escape(fig_num_str)}\b',
            ]
        
        referenced_blocks = []
        
        reference_type = "equation" if is_equation_num else "figure"
        logger.debug(f"Searching for {reference_type} {figure_number} references in {len(all_text_blocks)} blocks")
        
        for block in all_text_blocks:
            text = block['text']
            
            # Skip if this is the caption itself
            if caption_text and text.strip() == caption_text.strip():
                continue
            
            # Check if this block mentions the figure
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    referenced_blocks.append(block)
                    logger.debug(f"  Found reference in block: {text[:80]}")
                    break
        
        logger.debug(f"  Total referenced blocks found: {len(referenced_blocks)}")

        
        if not referenced_blocks:
            return ("", [])
        
        # Extract sentences containing the reference
        referenced_sentences = []
        positions = []
        
        for block in referenced_blocks:
            text = block['text']
            
            # =====================================================================
            # NEW: Extract FULL BLOCK/PARAGRAPH for better context
            # Instead of splitting into sentences, use the entire block text
            # =====================================================================
            
            # Check if this block mentions the figure
            has_figure_ref = False
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    has_figure_ref = True
                    break
            
            if not has_figure_ref:
                continue
            
            # Use the FULL block text
            clean_text = text.strip()
            
            # Quality filters for the entire block
            if len(clean_text) < 30:  # Too short
                continue
            
            words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text)
            if len(words) < 5:  # Need substantial content
                continue
            
            # Skip ONLY pure references with no explanation
            if len(clean_text) < 25:
                if re.match(r'^(See|Refer to|In|As shown in)\s+Fig\.?\s*\d+\s*[.!?]?$', clean_text, re.IGNORECASE):
                    continue
            
            # ACCEPT the entire block
            referenced_sentences.append(clean_text)
            positions.append((
                block['char_start'],
                block['char_end']
            ))
        
        if not referenced_sentences:
            # Fallback: Try block-level extraction for poorly formatted text
            logger.debug("  No sentences found, trying block-level extraction...")
            for block in referenced_blocks[:5]:  # Try first 5 blocks
                text = block['text'].strip()
                if len(text) >= 25 and len(re.findall(r'\b[a-zA-Z]{3,}\b', text)) >= 3:
                    referenced_sentences.append(text)
                    positions.append((block['char_start'], block['char_end']))
            
            if not referenced_sentences:
                return ("", [])
        
        # =====================================================================
        # Find SINGLE BEST block (NO concatenation - must maintain position alignment)
        # Prioritize blocks that are already long enough (150+ chars)
        # =====================================================================
        
        # Separate into long and short blocks
        long_blocks = [(text, pos) for text, pos in zip(referenced_sentences, positions) if len(text) >= 150]
        all_blocks = list(zip(referenced_sentences, positions))
        
        # Prefer long blocks if available
        if long_blocks:
            # Sort long blocks by length (longest first)
            long_blocks.sort(key=lambda x: len(x[0]), reverse=True)
            best_text, best_position = long_blocks[0]
            logger.debug(f"  Found long block: {len(best_text)} chars")
        else:
            # No long blocks, take the longest available (even if short)
            all_blocks.sort(key=lambda x: len(x[0]), reverse=True)
            best_text, best_position = all_blocks[0]
            logger.debug(f"  Best available block: {len(best_text)} chars (no long blocks found)")
        
        # Limit to 800 chars maximum (increased from 600 for more context)
        if len(best_text) > 800:
            truncated = best_text[:800]
            last_period = truncated.rfind('.')
            if last_period > 200:
                best_text = truncated[:last_period + 1]
            else:
                last_space = truncated.rfind(' ')
                if last_space > 200:
                    best_text = truncated[:last_space] + "..."
                else:
                    best_text = truncated + "..."
        
        logger.debug(f"  Final reference: {len(best_text)} chars (single block, no concatenation)")
        
        # Return single reference with its position
        return (best_text, [best_position])
    
    
    def _find_surrounding_text(
        self, image_bbox: Tuple[float, float, float, float],
        text_blocks: List[Dict],
        caption_text: str = "",
        original_caption_text: str = ""
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Find text surrounding image (above, below, left, right).
        
        Args:
            image_bbox: Bounding box of the image
            text_blocks: List of text blocks from the page
            caption_text: Cleaned caption text (without "Figure X:" prefix)
            original_caption_text: Original caption text (with "Figure X:" prefix)
        
        Returns: (surrounding_text, (char_start, char_end))
        """
        if not image_bbox:
            return ("", (0, 0))
        
        image_x0, image_y0, image_x1, image_y1 = image_bbox
        image_center_x = (image_x0 + image_x1) / 2
        image_center_y = (image_y0 + image_y1) / 2
        
        nearby_blocks = []
        
        for block in text_blocks:
            # Skip if this is the caption (check both cleaned and original versions)
            if caption_text and block['text'] == caption_text:
                continue
            if original_caption_text and block['text'] == original_caption_text:
                continue
            
            block_x0, block_y0, block_x1, block_y1 = block['bbox']
            block_center_x = (block_x0 + block_x1) / 2
            block_center_y = (block_y0 + block_y1) / 2
            
            # Calculate distance from image center
            distance = math.sqrt(
                (block_center_x - image_center_x)**2 + 
                (block_center_y - image_center_y)**2
            )
            
            if distance <= self.surrounding_text_radius:
                nearby_blocks.append({
                    'text': block['text'],
                    'distance': distance,
                    'char_start': block['char_start'],
                    'char_end': block['char_end']
                })
        
        if not nearby_blocks:
            return ("", (0, 0))
        
        # Sort by distance
        nearby_blocks.sort(key=lambda x: x['distance'])
        
        # TIER 1: Try strict filtering for high-quality text
        valid_blocks = []
        for block in nearby_blocks[:15]:  # Check top 15 candidates
            text = block['text']
            
            # Strict filters
            if len(text) < 30:
                continue
            
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            if len(words) < 3:
                continue
            
            if self._is_matrix_text(text):
                continue
            
            math_chars = len(re.findall(r'[θφπ∑∏∫=+\-×÷√|⟩⟨]', text))
            if math_chars > len(text) * 0.3:  # > 30% math
                continue
            
            valid_blocks.append(block)
            
            # Collect up to 5 good blocks for richer context
            if len(valid_blocks) >= 5:
                break
        
        # TIER 2: If strict filtering found nothing, try relaxed filtering
        if not valid_blocks:
            for block in nearby_blocks[:15]:
                text = block['text']
                
                # Relaxed filters
                if len(text) < 20:  # Shorter minimum
                    continue
                
                words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
                if len(words) < 1:  # At least 1 word
                    continue
                
                if self._is_matrix_text(text):
                    continue
                
                math_chars = len(re.findall(r'[θφπ∑∏∫=+\-×÷√|⟩⟨]', text))
                if math_chars > len(text) * 0.5:  # > 50% math (more lenient)
                    continue
                
                valid_blocks.append(block)
                
                if len(valid_blocks) >= 3:
                    break
        
        # TIER 3: Last resort - take any nearby text that's not pure matrix
        if not valid_blocks:
            for block in nearby_blocks[:10]:
                text = block['text']
                
                if len(text) < 15:
                    continue
                
                if not self._is_matrix_text(text):
                    valid_blocks.append(block)
                    
                    if len(valid_blocks) >= 2:
                        break
        
        if not valid_blocks:
            return ("", (0, 0))
        
        # =====================================================================
        # Pick SINGLE BEST surrounding block (NO concatenation)
        # Strategy: Longest block closest to the image
        # =====================================================================
        
        # Sort by length (longest first = most context)
        valid_blocks.sort(key=lambda b: len(b['text']), reverse=True)
        
        # Pick the single best block (longest)
        best_block = valid_blocks[0]
        surrounding_text = best_block['text']
        
        # Limit to 600 characters if needed
        if len(surrounding_text) > 600:
            truncated = surrounding_text[:600]
            last_period = truncated.rfind('.')
            if last_period > 100:
                surrounding_text = truncated[:last_period + 1]
            else:
                surrounding_text = truncated + "..."
        
        # Return single block's exact position
        return (surrounding_text, (best_block['char_start'], best_block['char_end']))
    
    def _extract_embedded_images(
        self, page: fitz.Page, page_num: int, arxiv_id: str, output_dir: Path
    ) -> List[Dict]:
        """Extract embedded raster images WITH bbox estimation."""
        figures = []
        
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                width, height = base_image["width"], base_image["height"]
                
                # Filter: Skip small images (use config values)
                min_width = self.config.get("detection.minWidth", 200)
                min_height = self.config.get("detection.minHeight", 150)
                
                if width < min_width or height < min_height:
                    continue
                
                # Filter: Skip very narrow or very tall images (aspect ratio check)
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    # Too narrow (< 0.2) or too tall (> 5.0) - not a meaningful circuit
                    continue
                
                # Filter: Skip portrait orientation (height > width)
                # Quantum circuits are ALWAYS landscape because wires run horizontally
                if height > width:
                    # Portrait orientation - cannot be a quantum circuit
                    continue
                
                # Filter: Skip matrices (visual detection)
                # Load image for matrix detection
                from PIL import Image
                import io
                pil_img = Image.open(io.BytesIO(image_bytes))
                img_array = np.array(pil_img.convert('L'))  # Convert to grayscale
                
                is_matrix, matrix_confidence = self.matrix_detector.is_matrix(img_array)
                if is_matrix:
                    # Matrix detected - filter it out
                    logger.debug(f"Matrix detected (confidence: {matrix_confidence:.2f}) - filtering")
                    continue
                
                # Save image
                output_path = output_dir / f"{arxiv_id}_p{page_num + 1}_embedded{img_index}.png"
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                
                # ESTIMATE BBOX for embedded images
                bbox = self._estimate_embedded_image_bbox(page, xref)
                
                figures.append({
                    'path': str(output_path),
                    'page': page_num + 1,
                    'method': 'embedded',
                    'size': (width, height),
                    'bbox': bbox  # Now has estimated bbox!
                })
                
            except Exception as e:
                logger.debug(f"Failed to extract embedded image: {e}")
        
        return figures
    
    def _estimate_embedded_image_bbox(
        self, page: fitz.Page, xref: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """Estimate bounding box for embedded image using image rectangles."""
        try:
            # Get all image rectangles on page
            image_rects = page.get_image_rects(xref)
            
            if image_rects:
                # Use first occurrence (most common case)
                rect = image_rects[0]
                return (rect.x0, rect.y0, rect.x1, rect.y1)
            
            # Fallback: search entire page for this image
            # This is slower but more thorough
            for img_info in page.get_images(full=True):
                if img_info[0] == xref:
                    # Try to get position from image list
                    # Note: This is a best-effort approach
                    break
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to estimate bbox for xref {xref}: {e}")
            return None
    
    def _extract_vector_regions(
        self, page: fitz.Page, page_num: int, arxiv_id: str, output_dir: Path
    ) -> List[Dict]:
        """Extract vector graphics regions."""
        drawings = page.get_drawings()
        if not drawings:
            return []
        
        # Group drawings into regions
        regions = self._group_drawings_into_regions(drawings, page.rect)
        
        figures = []
        for i, (x0, y0, x1, y1) in enumerate(regions):
            try:
                # Render region
                clip_rect = fitz.Rect(x0, y0, x1, y1)
                zoom = self.dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
                
                # Save
                output_path = output_dir / f"{arxiv_id}_p{page_num + 1}_vector{i}.png"
                pix.save(output_path)
                
                figures.append({
                    'path': str(output_path),
                    'page': page_num + 1,
                    'method': 'vector',
                    'size': (pix.width, pix.height),
                    'bbox': (x0, y0, x1, y1)
                })
                
            except Exception as e:
                logger.debug(f"Failed to extract vector region: {e}")
        
        return figures
    
    def _extract_rendered_figures(
        self, page: fitz.Page, page_num: int, arxiv_id: str, output_dir: Path
    ) -> List[Dict]:
        """Extract rendered regions with strict text/math filtering."""
        # Render page
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Find figure regions with strict filtering
        regions = self._find_figure_regions_strict(gray)
        
        figures = []
        for i, (x0, y0, x1, y1) in enumerate(regions):
            try:
                # Crop region
                cropped = img[y0:y1, x0:x1]
                
                if cropped.shape[0] < 100 or cropped.shape[1] < 150:
                    continue
                
                # Filter: Skip matrices (visual detection)
                # Convert to grayscale for matrix detection
                gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
                is_matrix, matrix_confidence = self.matrix_detector.is_matrix(gray_cropped)
                if is_matrix:
                    logger.debug(f"Matrix detected in rendered region (confidence: {matrix_confidence:.2f}) - filtering")
                    continue
                
                # Save
                output_path = output_dir / f"{arxiv_id}_p{page_num + 1}_rendered{i}.png"
                cv2.imwrite(str(output_path), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                
                # Convert pixel coords to PDF coords
                scale = 72.0 / self.dpi
                bbox = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
                
                figures.append({
                    'path': str(output_path),
                    'page': page_num + 1,
                    'method': 'rendered',
                    'size': (cropped.shape[1], cropped.shape[0]),
                    'bbox': bbox
                })
                
            except Exception as e:
                logger.debug(f"Failed to extract rendered region: {e}")
        
        return figures
    
    def _group_drawings_into_regions(
        self, drawings: list, page_rect: fitz.Rect
    ) -> List[Tuple[float, float, float, float]]:
        """Group nearby drawings into regions."""
        if not drawings:
            return []
        
        # Get bounding boxes
        bboxes = []
        for drawing in drawings:
            if 'rect' in drawing:
                rect = drawing['rect']
                w = rect.x1 - rect.x0
                h = rect.y1 - rect.y0
                if w > 5 and h > 5:  # Skip tiny elements
                    bboxes.append([rect.x0, rect.y0, rect.x1, rect.y1])
        
        if not bboxes:
            return []
        
        # Merge nearby boxes
        merged = self._merge_nearby_boxes(bboxes, self.merge_threshold)
        
        # Filter by size
        regions = []
        for box in merged:
            x0, y0, x1, y1 = box
            width = x1 - x0
            height = y1 - y0
            area = width * height
            
            # Size filter (use config values)
            min_width = self.config.get("detection.minWidth", 200)
            min_height = self.config.get("detection.minHeight", 150)
            min_area = min_width * min_height  # Minimum area
            
            if width >= min_width and height >= min_height and area > min_area:
                # Aspect ratio check - avoid very narrow or very tall regions
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                    continue  # Skip very narrow/tall regions
                
                # Portrait orientation check - quantum circuits are landscape
                if height > width:
                    continue  # Skip portrait regions
                
                # Not full page
                if width < page_rect.width * 0.95 and height < page_rect.height * 0.8:
                    regions.append((x0, y0, x1, y1))
        
        return regions
    
    def _merge_nearby_boxes(
        self, boxes: list, threshold: float
    ) -> list:
        """Merge boxes within threshold distance."""
        if not boxes:
            return []
        
        boxes = [list(b) for b in boxes]
        
        changed = True
        while changed:
            changed = False
            new_boxes = []
            used = set()
            
            for i, box1 in enumerate(boxes):
                if i in used:
                    continue
                
                merged_box = list(box1)
                for j, box2 in enumerate(boxes):
                    if i == j or j in used:
                        continue
                    
                    # Calculate gap
                    h_gap = max(0, max(merged_box[0], box2[0]) - min(merged_box[2], box2[2]))
                    v_gap = max(0, max(merged_box[1], box2[1]) - min(merged_box[3], box2[3]))
                    
                    if h_gap <= threshold and v_gap <= threshold:
                        # Merge
                        merged_box[0] = min(merged_box[0], box2[0])
                        merged_box[1] = min(merged_box[1], box2[1])
                        merged_box[2] = max(merged_box[2], box2[2])
                        merged_box[3] = max(merged_box[3], box2[3])
                        used.add(j)
                        changed = True
                
                used.add(i)
                new_boxes.append(merged_box)
            
            boxes = new_boxes
        
        return boxes
    
    def _find_figure_regions_strict(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find figure regions with strict text/math filtering."""
        height, width = gray.shape
        
        # Morphological operations
        _, light_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        content_mask = cv2.bitwise_not(light_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
        dilated = cv2.dilate(content_mask, kernel, iterations=2)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel2)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Size filter
            if not (w >= 200 and h >= 100 and 
                    w < width * 0.9 and h < height * 0.7 and 
                    w * h > 20000):
                continue
            
            # Extract ROI
            roi = gray[y:y+h, x:x+w]
            
            # FILTER 1: Text paragraph
            if self._is_text_region(roi):
                continue
            
            # FILTER 2: Smart Math Formula Filter (TWO-STAGE)
            # Stage 1: Check if it has circuit structure
            has_circuit = self._has_circuit_structure(roi)
            
            if has_circuit:
                # Has circuit structure → KEEP IT (even if it has math notation)
                # This preserves circuits with gate labels like θ, φ, π/4, etc.
                pass
            else:
                # No circuit structure → Apply math formula filter
                if self._is_math_formula(roi):
                    # Pure math equation → FILTER IT OUT
                    continue
            
            # Passed filters - keep it           
            # STRUCTURE CHECK
            if not self._has_structured_content(roi):
                continue
            
            # Passed all filters
            regions.append((x, y, x+w, y+h))
        
        return regions
    
    def _is_text_region(self, roi: np.ndarray) -> bool:
        """Detect if region is text."""
        _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return True  # Empty
        
        # Text has many small contours
        areas = [cv2.contourArea(c) for c in contours]
        avg_area = np.mean(areas)
        
        if len(contours) > 50 and avg_area < 150:
            return True  # Text!
        
        # Check density
        density = np.sum(binary > 0) / roi.size
        if 0.1 < density < 0.35 and len(contours) > 30:
            return True  # Text!
        
        return False
    
    def _is_math_formula(self, roi: np.ndarray) -> bool:
        """Detect and filter math formulas (ENHANCED ACCURACY)."""
        height, width = roi.shape[:2]
        
        # Rule 1: Very short regions (inline equations)
        if height < 80:
            return True
        
        # Rule 2: Wide and short (display equations)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 2.5 and height < 150:
            return True
        
        # Rule 3: Very wide and relatively short (equation blocks)
        if aspect_ratio > 3.0 and height < 250:
            return True
        
        # Get binary image for analysis
        _, binary = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return True  # Empty
        
        # Rule 4: Check for many tiny symbols (math notation)
        tiny_contours = sum(1 for c in contours if cv2.contourArea(c) < 50)
        if tiny_contours > 20 and height < 200:
            return True
        
        # Rule 5: Check symbol density (math has sparse symbols)
        areas = [cv2.contourArea(c) for c in contours]
        avg_area = np.mean(areas)
        if len(contours) > 15 and avg_area < 100 and height < 300:
            return True  # Many small symbols = likely math
        
        # Rule 6: Check horizontal distribution (equations are horizontally spread)
        if len(contours) > 10:
            x_positions = [cv2.boundingRect(c)[0] for c in contours]
            x_spread = max(x_positions) - min(x_positions)
            if x_spread > width * 0.7 and height < 250:
                # Horizontally distributed symbols = equation
                return True
        
        # Rule 7: Check for equation-like patterns (symbols in a line)
        if len(contours) > 8 and aspect_ratio > 2.0:
            y_positions = [cv2.boundingRect(c)[1] for c in contours]
            y_variance = np.var(y_positions)
            if y_variance < 100:  # Symbols aligned horizontally
                return True
        
        return False
    
    def _has_circuit_structure(self, roi: np.ndarray) -> bool:
        """
        Detect if image has quantum circuit structure.
        
        This is used to distinguish quantum circuits (which may contain math notation)
        from pure mathematical equations.
        
        Circuit indicators:
        1. Multiple horizontal lines (quantum wires)
        2. Rectangular boxes (gates)
        3. Grid-like pattern
        
        Returns:
            True if circuit structure detected, False otherwise
        """
        height, width = roi.shape[:2]
        
        # Detect edges
        edges = cv2.Canny(roi, 50, 150)
        
        # 1. DETECT HORIZONTAL LINES (quantum wires)
        # Use Hough Line Transform to find lines
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=40,
            minLineLength=int(width * 0.25),  # At least 25% of width
            maxLineGap=15
        )
        
        horizontal_line_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # Horizontal if angle close to 0 or 180
                if angle < 15 or angle > 165:
                    horizontal_line_count += 1
        
        # 2. DETECT RECTANGULAR BOXES (gates)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_boxes = 0
        for contour in contours:
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if it's a rectangle (4 corners)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Filter for gate-like boxes (not too thin, reasonable size)
                if 0.3 < aspect_ratio < 5.0 and area > 100:
                    rectangular_boxes += 1
        
        # 3. DECISION LOGIC
        # Strong circuit indicators
        if horizontal_line_count >= 2 and rectangular_boxes >= 1:
            # Multiple horizontal lines + boxes = quantum circuit!
            return True
        
        if horizontal_line_count >= 3:
            # Many horizontal lines = likely circuit wires
            return True
        
        if rectangular_boxes >= 4 and horizontal_line_count >= 1:
            # Many boxes + at least one wire = circuit
            return True
        
        # 4. CHECK FOR GRID PATTERN (additional check)
        # Circuits often have regular spacing
        if horizontal_line_count >= 2 and rectangular_boxes >= 2:
            # Even weak grid pattern suggests circuit
            return True
        
        return False

    
    def _has_structured_content(self, roi: np.ndarray) -> bool:
        """Check for lines/boxes (figures have structure)."""
        edges = cv2.Canny(roi, 50, 150)
        
        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None and len(lines) > 5:
            return True  # Has lines = figure
        
        # Rectangle detection
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = sum(1 for c in contours 
                        if len(cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)) == 4)
        
        if rectangles > 3:
            return True  # Has boxes = figure
        
        return False
    
    def _deduplicate_figures(self, figures: List[Dict]) -> List[Dict]:
        """Remove duplicate figures based on page and bbox overlap (STRICT)."""
        # Sort by size (keep larger versions)
        figures = sorted(figures, 
                        key=lambda f: f['size'][0] * f['size'][1], 
                        reverse=True)
        
        unique = []
        for fig in figures:
            is_duplicate = False
            
            for existing in unique:
                if fig['page'] == existing['page']:
                    if fig['bbox'] and existing['bbox']:
                        iou = self._calculate_iou(fig['bbox'], existing['bbox'])
                        # STRICTER: 40% overlap = duplicate (was 60%)
                        if iou > 0.4:
                            is_duplicate = True
                            break
                    elif not fig['bbox'] and not existing['bbox']:
                        # Both embedded images without bbox - check if same size
                        if (abs(fig['size'][0] - existing['size'][0]) < 10 and
                            abs(fig['size'][1] - existing['size'][1]) < 10):
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                unique.append(fig)
        
        return unique
    
    def _calculate_iou(
        self, bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Calculate Intersection over Union of two bboxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0
        
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _create_image_info_with_metadata(
        self, figures: List[Dict], arxiv_id: str,
        page_text_blocks: Dict[int, List[Dict]],
        pdf_data: PdfData
    ) -> List[ImageInfo]:
        """Convert figure dicts to ImageInfo objects WITH METADATA."""
        images = []
        
        # Flatten all text blocks from all pages for figure reference search
        all_text_blocks = []
        for page_blocks in page_text_blocks.values():
            all_text_blocks.extend(page_blocks)
        
        for idx, fig in enumerate(figures):
            page_num = fig['page'] - 1  # Convert to 0-indexed
            text_blocks = page_text_blocks.get(page_num, [])
            
            # Extract metadata
            caption_info = None
            figure_number = None
            caption_text = ""
            caption_positions = (0, 0)
            referenced_text = ""
            referenced_positions = []
            surrounding_text = ""
            surrounding_positions = (0, 0)
            
            if fig['bbox']:
                # Find caption
                caption_info = self._find_caption(fig['bbox'], text_blocks)
                if caption_info:
                    # Extract figure number first (from full caption)
                    figure_number = self._extract_figure_number(caption_info['text'])
                    
                    # Remove "Figure X:" prefix from caption text
                    caption_text, char_start, char_end = self._remove_figure_prefix(
                        caption_info['text'],
                        caption_info['char_start'],
                        caption_info['char_end']
                    )
                    caption_positions = (char_start, char_end)
                
                # FALLBACK: If no figure number, try spatial equation detection
                if not figure_number:
                    eq_num, eq_text = self._find_equation_number(fig['bbox'], text_blocks)
                    if eq_num:
                        # Use just the number as integer
                        try:
                            figure_number = int(eq_num.split('.')[0])
                            is_equation = True  # Track that this is equation-based
                        except:
                            figure_number = None
                            is_equation = False
                        
                        # Use equation text as caption if no caption
                        if eq_text and not caption_text:
                            caption_text = eq_text
                            caption_positions = (0, len(eq_text))
                            logger.debug(f"  Using equation text as caption: '{eq_text}'")
                        
                        logger.debug(f"  Spatial equation: num={figure_number}")
                else:
                    is_equation = False
                
                # Always extract surrounding text (spatial proximity)
                surrounding_text, surrounding_positions = self._find_surrounding_text(
                    fig['bbox'], text_blocks, caption_text,
                    original_caption_text=caption_info['text'] if caption_info else ""
                )
                
                # Also try to find referenced text (document-wide)
                if figure_number:
                    # For equations, pass as "Eq.15" to trigger equation pattern search
                    search_key = f"Eq.{figure_number}" if is_equation else str(figure_number)
                    
                    referenced_text, referenced_positions_list = self._find_figure_references(
                        search_key,
                        all_text_blocks,
                        caption_text
                    )
                    
                    # Keep the list of positions directly (each is a tuple (start, end))
                    # These are now document-wide positions due to the offset we added earlier
                    referenced_positions = referenced_positions_list if referenced_positions_list else []

            
            # Create bbox object
            bbox = BoundingBox(
                x0=fig['bbox'][0],
                y0=fig['bbox'][1],
                x1=fig['bbox'][2],
                y1=fig['bbox'][3]
            ) if fig['bbox'] else None
            
            # Debug logging for text extraction
            logger.debug(f"Image {idx + 1} (Figure {figure_number if figure_number else 'N/A'}):")
            logger.debug(f"  Caption: {caption_text[:80] if caption_text else 'EMPTY'}")
            logger.debug(f"  Surrounding: {surrounding_text[:80] if surrounding_text else 'EMPTY'}")
            logger.debug(f"  Referenced: {referenced_text[:80] if referenced_text else 'EMPTY'}")
            
            image_info = ImageInfo(
                tempPath=fig['path'],
                arxivId=arxiv_id,
                pageNumber=fig['page'],
                imageIndex=idx,
                width=fig['size'][0],
                height=fig['size'][1],
                extractionMethod=fig['method'],
                bbox=bbox,
                captionText=caption_text,
                surroundingText=surrounding_text,  # Spatial proximity text
                referencedText=referenced_text,  # Document-wide referenced text
                captionPositions=caption_positions,
                surroundingPositions=surrounding_positions,
                referencedPositions=referenced_positions,
                figureNumber=figure_number
            )
            
            images.append(image_info)
        
        return images
    
    def _extract_pdf_data(
        self, doc: fitz.Document, title: str, abstract: str
    ) -> PdfData:
        """Extract PDF-level data."""
        # Extract text from all pages
        page_texts = []
        for page in doc:
            page_texts.append(page.get_text())
        
        full_text = "\n".join(page_texts)
        
        # Calculate page boundaries
        page_boundaries = []
        char_pos = 0
        for page_text in page_texts:
            start = char_pos
            end = char_pos + len(page_text)
            page_boundaries.append((start, end))
            char_pos = end + 1
        
        return PdfData(
            fullText=full_text,
            pageTexts=page_texts,
            pageBoundaries=page_boundaries,
            title=title,  # From arXiv API
            abstract=abstract,  # From arXiv API
            totalPages=len(doc)
        )

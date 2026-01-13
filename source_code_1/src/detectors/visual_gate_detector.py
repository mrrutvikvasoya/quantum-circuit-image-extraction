"""
Visual Quantum Gate Detector using Contour Detection + EasyOCR
Replaces the old Tesseract-based approach with visual box detection.
"""
import cv2
import numpy as np
import re
from typing import List, Dict, Set, Tuple
from pathlib import Path

# Lazy import for EasyOCR
_easyocr_reader = None


class VisualQuantumGateDetector:
    """Visual gate detector using contour detection and EasyOCR"""
    
    # Comprehensive gate lists (80+ gates) - DETECT ALL
    HIGH_CONFIDENCE_GATES = {
        # Two-qubit controlled gates
        'CNOT', 'CX', 'CY', 'CZ', 'CH', 'CS', 'CT', 'CP', 'CPHASE',
        'CRX', 'CRY', 'CRZ', 'CU', 'CU1', 'CU2', 'CU3',
        
        # SWAP family
        'SWAP', 'ISWAP', 'SQRTSWAP',
        
        # Three+ qubit gates
        'TOFFOLI', 'CCX', 'CCNOT', 'CCZ', 'CSWAP', 'FREDKIN',
        'MCX', 'MCZ', 'C3X', 'C4X',
        
        # Two-qubit rotation gates
        'RXX', 'RYY', 'RZZ', 'RZX', 'XX', 'YY', 'ZZ',
        
        # Special operations
        'DCX', 'ECR', 'BARRIER', 'MEASURE', 'RESET', 'HADAMARD'
    }
    
    MEDIUM_CONFIDENCE_GATES = {
        # Rotation gates
        'RX', 'RY', 'RZ', 'Rx', 'Ry', 'Rz',
        
        # Universal gates
        'U', 'U1', 'U2', 'U3',
        
        # Phase gates
        'P', 'PHASE',
        
        # Special single-qubit gates
        'SX', 'SQRTX', 'SDG', 'TDG', 'V', 'VDG'
    }
    
    LOW_CONFIDENCE_GATES = {
        # Pauli gates
        'H', 'X', 'Y', 'Z',
        
        # Phase gates
        'S', 'T',
        
        # Identity
        'I', 'ID',
        
        # Measurement
        'M'
    }
    
    ALL_GATES = HIGH_CONFIDENCE_GATES | MEDIUM_CONFIDENCE_GATES | LOW_CONFIDENCE_GATES
    
    # False positives to filter
    FALSE_POSITIVE_PATTERNS = {
        'IN', 'OUT', 'IF', 'OR', 'AND', 'NOT',
        'TO', 'AT', 'IT', 'IS', 'AS', 'ON', 'OF', 'BY',
        'FIG', 'PAGE', 'REF', 'EQ', 'TABLE',
        'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'J', 'K', 'L', 'N', 'O', 'Q', 'W'
    }
    
    # Quantum-specific gates (for JSON validation)
    # Only these gates will be included in final JSON output
    QUANTUM_SPECIFIC_GATES = {
        # Core quantum gates
        'H', 'HADAMARD',
        'X', 'Y', 'Z',  # Pauli gates
        'S', 'T',  # Phase gates
        
        # Controlled gates
        'CNOT', 'CX', 'CY', 'CZ', 'CH', 'CS', 'CT',
        
        # Multi-qubit gates
        'SWAP', 'ISWAP', 'TOFFOLI', 'CCX', 'FREDKIN', 'CSWAP',
        
        # Rotation gates
        'RX', 'RY', 'RZ', 'CRX', 'CRY', 'CRZ',
        
        # Universal gates
        'U', 'U1', 'U2', 'U3',
        
        # Special gates
        'MEASURE', 'M', 'RESET', 'BARRIER'
    }
    
    # Control gates (indicate multi-qubit circuits)
    CONTROL_GATES = {
        'CNOT', 'CX', 'CZ', 'CY', 'CH', 'CS', 'CT',
        'TOFFOLI', 'CCX', 'CCNOT', 'CCZ', 'CSWAP', 'FREDKIN'
    }
    
    def __init__(self, use_gpu: bool = False, config: dict = None):
        """
        Initialize visual gate detector.
        
        Args:
            use_gpu: Use GPU for EasyOCR if available
            config: Configuration dict with thresholds
        """
        self.use_gpu = use_gpu
        self.reader = None  # Lazy load
        
        # Load config or use defaults
        if config:
            visual_cfg = config.get('gateDetection', {}).get('visual', {})
            self.min_area = visual_cfg.get('minArea', 400)
            self.max_area = visual_cfg.get('maxArea', 50000)
            self.min_aspect = visual_cfg.get('minAspectRatio', 0.3)
            self.max_aspect = visual_cfg.get('maxAspectRatio', 4.0)
            self.thresholds = visual_cfg.get('thresholds', [150, 180, 200])
        else:
            self.min_area = 400
            self.max_area = 50000
            self.min_aspect = 0.3
            self.max_aspect = 4.0
            self.thresholds = [150, 180, 200]
    
    def _init_easyocr(self):
        """Lazy initialization of EasyOCR"""
        global _easyocr_reader
        if _easyocr_reader is None:
            try:
                import easyocr
                print("  → Loading EasyOCR model (first run downloads ~100MB)...")
                _easyocr_reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
                print("  ✓ EasyOCR ready")
            except ImportError:
                print("  ⚠ EasyOCR not installed. Install with: pip install easyocr")
                _easyocr_reader = False
            except Exception as e:
                print(f"  ⚠ EasyOCR initialization failed: {e}")
                _easyocr_reader = False
        
        self.reader = _easyocr_reader if _easyocr_reader is not False else None
    
    def _find_gate_regions(self, gray: np.ndarray, threshold_value: int) -> List[Tuple[int, int, int, int]]:
        """
        Find rectangular gate regions using contour detection.
        
        Args:
            gray: Grayscale image
            threshold_value: Binary threshold value
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        # Binary threshold
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        gate_regions = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = w / h if h > 0 else 0
            
            # Filter for gate-like boxes
            if (self.min_aspect < aspect < self.max_aspect and 
                self.min_area < area < self.max_area and 
                w > 20 and h > 20):
                
                # Check if roughly rectangular
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) >= 4:
                    gate_regions.append((x, y, w, h))
        
        return gate_regions
    
    def _remove_nested(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove boxes that are inside other boxes."""
        filtered = []
        for box in boxes:
            x1, y1, w1, h1 = box
            is_nested = False
            for other in boxes:
                if box == other:
                    continue
                x2, y2, w2, h2 = other
                # Check if box is inside other
                if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
                    is_nested = True
                    break
            if not is_nested:
                filtered.append(box)
        return filtered
    
    def _remove_duplicates(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Remove duplicate/overlapping boxes."""
        if not boxes:
            return []
        
        # Sort by area (largest first)
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        
        filtered = []
        for box in boxes:
            x1, y1, w1, h1 = box
            is_duplicate = False
            
            for existing in filtered:
                x2, y2, w2, h2 = existing
                
                # Calculate overlap
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                # If >50% overlap, consider duplicate
                area1 = w1 * h1
                if overlap_area > 0.5 * area1:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(box)
        
        return filtered
    
    def _normalize_gate_name(self, text: str) -> str:
        """Normalize OCR output to standard gate names."""
        if not text:
            return ''
        
        text = text.upper().strip()
        
        # Handle special characters
        text = text.replace('†', 'DG').replace('√', 'SQRT')
        text = text.replace('DAG', 'DG').replace('DAGGER', 'DG')
        
        # Remove parameters: Rx(θ) → RX
        text = re.sub(r'\([^)]+\)', '', text).strip()
        
        # Remove common OCR errors
        text = text.replace('0', 'O').replace('1', 'I')
        
        # Remove spaces
        text = text.replace(' ', '')
        
        return text
    
    
    def _preprocess_roi(self, roi: np.ndarray) -> List[np.ndarray]:
        """
        Preprocess ROI with 3 best strategies (optimized for speed).
        
        Returns list of preprocessed images to try.
        """
        preprocessed = []
        
        # Strategy 1: Original (fastest, often works)
        preprocessed.append(roi)
        
        # Strategy 2: CLAHE - Contrast enhancement (best for low contrast)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(roi)
        preprocessed.append(enhanced)
        
        # Strategy 3: Adaptive threshold (best for varying lighting)
        adaptive = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        preprocessed.append(adaptive)
        
        # Removed: Denoising (slow, minimal benefit)
        # Removed: Morphological (slow, minimal benefit)
        
        return preprocessed
    
    def _ocr_region(self, gray: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """
        OCR a specific region with multiple preprocessing strategies.
        
        Args:
            gray: Grayscale image
            region: (x, y, w, h) bounding box
            
        Returns:
            Detected text
        """
        if self.reader is None:
            return ''
        
        x, y, w, h = region
        
        # Add more padding for better OCR
        pad = 10  # Increased from 5
        y1 = max(0, y - pad)
        y2 = min(gray.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(gray.shape[1], x + w + pad)
        
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return ''
        
        # Try multiple preprocessing strategies
        preprocessed_images = self._preprocess_roi(roi)
        
        best_result = ''
        max_confidence = 0
        
        for prep_img in preprocessed_images:
            try:
                # EasyOCR with confidence scores
                results = self.reader.readtext(prep_img, detail=1, paragraph=False)
                
                if results:
                    # Get result with highest confidence
                    for (bbox, text, conf) in results:
                        if conf > max_confidence:
                            max_confidence = conf
                            best_result = text.strip()
            except Exception:
                continue
        
        return best_result

    
    def detect_gates(self, image_path: str, verbose: bool = False) -> Dict:
        """
        Detect quantum gates using visual detection + OCR.
        
        Args:
            image_path: Path to circuit image
            verbose: Print detailed information
            
        Returns:
            Dictionary with detection results
        """
        if verbose:
            print(f"\nProcessing: {image_path}")
        
        # Initialize OCR if needed
        if self.reader is None:
            self._init_easyocr()
        
        if self.reader is None:
            return {
                'success': False,
                'error': 'EasyOCR not available',
                'gates': [],
                'confidence_scores': {}
            }
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return {
                'success': False,
                'error': 'Failed to load image',
                'gates': [],
                'confidence_scores': {}
            }
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find gate regions with multiple thresholds
        if verbose:
            print(f"  → Finding gate regions with thresholds: {self.thresholds}")
        
        all_regions = []
        for thresh_val in self.thresholds:
            regions = self._find_gate_regions(gray, thresh_val)
            all_regions.extend(regions)
        
        # Remove nested and duplicate boxes
        gate_regions = self._remove_nested(all_regions)
        gate_regions = self._remove_duplicates(gate_regions)
        
        # Sort: top-to-bottom, then left-to-right
        gate_regions.sort(key=lambda b: (b[1] // 30, b[0]))
        
        if verbose:
            print(f"  → Found {len(gate_regions)} potential gate regions")
        
        # OCR each region
        gates_dict = {}  # gate -> count
        
        for region in gate_regions:
            raw_text = self._ocr_region(gray, region)
            
            if not raw_text or len(raw_text) > 20:  # Filter noise
                continue
            
            # Normalize
            gate_name = self._normalize_gate_name(raw_text)
            
            # Validate
            if gate_name in self.FALSE_POSITIVE_PATTERNS:
                continue
            
            if gate_name in self.ALL_GATES:
                gates_dict[gate_name] = gates_dict.get(gate_name, 0) + 1
        
        # Create confidence scores
        confidence_scores = {}
        for gate, count in gates_dict.items():
            if gate in self.HIGH_CONFIDENCE_GATES:
                conf_level = 'high'
            elif gate in self.MEDIUM_CONFIDENCE_GATES:
                conf_level = 'medium'
            else:
                conf_level = 'low'
            
            confidence_scores[gate] = {
                'count': count,
                'confidence': conf_level
            }
        
        # All detected gates (for validation)
        all_gates_list = sorted(list(gates_dict.keys()))
        
        # Filter for JSON output: only quantum-specific gates
        quantum_gates_list = sorted([
            gate for gate in gates_dict.keys() 
            if gate in self.QUANTUM_SPECIFIC_GATES
        ])
        
        # Validate that gates are quantum-specific (using ALL detected gates)
        is_valid, validation_confidence, validation_reason = self.validate_quantum_gates(all_gates_list)
        
        if verbose:
            print(f"  ✓ Detected {len(all_gates_list)} gates total: {all_gates_list}")
            print(f"  → Quantum-specific gates for JSON: {quantum_gates_list}")
            print(f"  → Validation: {validation_reason} (confidence: {validation_confidence:.2f})")
        
        return {
            'success': True,
            'gates': quantum_gates_list,  # Only quantum-specific gates in JSON
            'gate_count': len(quantum_gates_list),
            'confidence_scores': {k: v for k, v in confidence_scores.items() if k in self.QUANTUM_SPECIFIC_GATES},
            'is_valid_quantum': is_valid,
            'validation_confidence': validation_confidence,
            'validation_reason': validation_reason
        }
    
    def validate_quantum_gates(self, gates_found: List[str]) -> Tuple[bool, float, str]:
        """
        Validate that detected gates are actually quantum gates.
        
        Args:
            gates_found: List of gate names detected
            
        Returns:
            Tuple of (is_valid, confidence, reason)
        """
        if len(gates_found) < 1:
            return False, 0.0, "no_gates_found"
        
        # Check 1: Has at least one quantum-specific gate
        has_quantum = any(g in self.QUANTUM_SPECIFIC_GATES for g in gates_found)
        if not has_quantum:
            return False, 0.3, "no_quantum_specific_gates"
        
        # Count unique gates for confidence scoring (but don't reject based on diversity)
        unique_gates = len(set(gates_found))
        
        # Check 2: Has control gate (higher confidence for multi-qubit circuits)
        has_control = any(g in self.CONTROL_GATES for g in gates_found)
        
        # Calculate confidence
        confidence = 0.6  # Base confidence
        
        if has_control:
            confidence += 0.2  # Multi-qubit circuit
        
        if unique_gates >= 3:
            confidence += 0.1  # Good diversity
        
        if len(gates_found) >= 4:
            confidence += 0.1  # Multiple gates
        
        # Determine reason
        if has_control and unique_gates >= 3:
            reason = "valid_multi_qubit_circuit"
        elif has_quantum and unique_gates >= 2:
            reason = "valid_quantum_circuit"
        else:
            reason = "borderline_circuit"
        
        return True, min(confidence, 1.0), reason

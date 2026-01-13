"""
Matrix Detector - Visual Detection of Matrix Representations

Matrices have distinctive visual characteristics:
1. Grid of numbers/symbols in rows and columns
2. Brackets on left and right: [ ] or ( )
3. Regular spacing between elements
4. NO horizontal lines (unlike circuits)
5. Uniform text size and spacing

This detector uses computer vision to identify these patterns.
"""

import cv2
import numpy as np
from typing import Tuple


class MatrixDetector:
    """
    Detect if an image is a matrix representation using visual analysis.
    
    Matrix characteristics:
    - Grid pattern of text elements
    - Vertical brackets on sides
    - Regular spacing
    - No horizontal lines (unlike circuits)
    """
    
    def __init__(self):
        """Initialize matrix detector."""
        pass
    
    def is_matrix(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image is a matrix representation.
        
        Args:
            image: Grayscale image (numpy array)
            
        Returns:
            Tuple of (is_matrix: bool, confidence: float)
        """
        height, width = image.shape[:2]
        
        # Score accumulator
        matrix_score = 0.0
        max_score = 6.0  # Total possible score
        
        # 1. CHECK FOR BRACKETS (strong indicator)
        has_brackets, bracket_score = self._detect_brackets(image)
        matrix_score += bracket_score * 2.0  # Weight: 2.0
        
        # 2. CHECK FOR GRID PATTERN
        has_grid, grid_score = self._detect_grid_pattern(image)
        matrix_score += grid_score * 1.5  # Weight: 1.5
        
        # 3. CHECK FOR LACK OF HORIZONTAL LINES
        # Matrices DON'T have horizontal lines (circuits DO)
        has_no_lines, no_line_score = self._check_no_horizontal_lines(image)
        matrix_score += no_line_score * 1.5  # Weight: 1.5
        
        # 4. CHECK FOR REGULAR SPACING
        has_spacing, spacing_score = self._detect_regular_spacing(image)
        matrix_score += spacing_score * 1.0  # Weight: 1.0
        
        # Calculate confidence
        confidence = matrix_score / max_score
        
        # Decision threshold (lowered to 0.4 to catch matrices with one bracket)
        # Single bracket (0.8) * weight (2.0) = 1.6 / 6.0 = 0.27
        # Single bracket + grid (1.0) * 1.5 = 1.6 + 1.5 = 3.1 / 6.0 = 0.52 → DETECTED
        # Single bracket + no lines (1.0) * 1.5 = 1.6 + 1.5 = 3.1 / 6.0 = 0.52 → DETECTED
        is_matrix = confidence > 0.4  # Lowered from 0.5
        
        return is_matrix, confidence
    
    def _detect_brackets(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect vertical brackets [ ] or ( ) on sides.
        
        Returns:
            (has_brackets, score)
        """
        height, width = image.shape[:2]
        
        # Look at left and right edges (10% of width)
        left_edge = image[:, :int(width * 0.1)]
        right_edge = image[:, int(width * 0.9):]
        
        # Detect vertical lines in edges
        edges_left = cv2.Canny(left_edge, 50, 150)
        edges_right = cv2.Canny(right_edge, 50, 150)
        
        # Count vertical pixels
        left_vertical = np.sum(edges_left > 0, axis=0)
        right_vertical = np.sum(edges_right > 0, axis=0)
        
        # Check if there are strong vertical lines
        # Lower threshold for better detection of thin brackets
        left_has_bracket = np.max(left_vertical) > height * 0.25  # Lowered from 0.3
        right_has_bracket = np.max(right_vertical) > height * 0.25
        
        if left_has_bracket and right_has_bracket:
            return True, 1.0  # Both brackets present - definitely matrix
        elif left_has_bracket or right_has_bracket:
            # Single bracket is still a STRONG indicator (circuits never have brackets)
            return True, 0.8  # Increased from 0.5 - single bracket is strong evidence
        else:
            return False, 0.0
    
    def _detect_grid_pattern(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect regular grid pattern of text elements.
        
        Returns:
            (has_grid, score)
        """
        # Threshold image
        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (text elements)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 4:
            return False, 0.0  # Too few elements
        
        # Get bounding boxes
        boxes = [cv2.boundingRect(c) for c in contours]
        
        # Check for regular spacing
        # Extract y-coordinates (rows)
        y_coords = [box[1] for box in boxes]
        y_coords_sorted = sorted(set(y_coords))
        
        # Check if y-coordinates form regular rows
        if len(y_coords_sorted) >= 2:
            y_diffs = np.diff(y_coords_sorted)
            y_variance = np.var(y_diffs) if len(y_diffs) > 0 else 1000
            
            # Low variance = regular rows
            if y_variance < 100:
                return True, 1.0
            elif y_variance < 300:
                return True, 0.5
        
        return False, 0.0
    
    def _check_no_horizontal_lines(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check that image does NOT have horizontal lines.
        Matrices don't have horizontal lines, circuits DO.
        
        Returns:
            (no_lines, score)
        """
        height, width = image.shape[:2]
        
        # Detect edges
        edges = cv2.Canny(image, 50, 150)
        
        # Detect horizontal lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=40,
            minLineLength=int(width * 0.2),
            maxLineGap=10
        )
        
        horizontal_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # Horizontal if angle close to 0 or 180
                if angle < 15 or angle > 165:
                    horizontal_count += 1
        
        # Matrices should have FEW or NO horizontal lines
        if horizontal_count == 0:
            return True, 1.0  # No lines = likely matrix
        elif horizontal_count <= 2:
            return True, 0.5  # Few lines = maybe matrix
        else:
            return False, 0.0  # Many lines = likely circuit
    
    def _detect_regular_spacing(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect regular spacing between elements.
        
        Returns:
            (has_spacing, score)
        """
        # Threshold image
        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 4:
            return False, 0.0
        
        # Get centers
        centers = []
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centers.append((cx, cy))
        
        if len(centers) < 4:
            return False, 0.0
        
        # Check spacing variance
        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]
        
        x_diffs = np.diff(sorted(x_coords))
        y_diffs = np.diff(sorted(y_coords))
        
        # Low variance in spacing = regular grid
        x_var = np.var(x_diffs) if len(x_diffs) > 0 else 1000
        y_var = np.var(y_diffs) if len(y_diffs) > 0 else 1000
        
        if x_var < 500 and y_var < 500:
            return True, 1.0
        elif x_var < 1000 or y_var < 1000:
            return True, 0.5
        
        return False, 0.0


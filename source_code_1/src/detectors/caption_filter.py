"""
Caption-Based Pre-Filter for Quantum Circuit Detection

Three-tier decision system:
1. WHITELIST → ACCEPT (circuit confirmed)
2. BLACKLIST → REJECT (not a circuit)
3. PASS → Continue to DINOv2 detection

Priority: If caption has BOTH whitelist and blacklist keywords, BLACKLIST wins.
"""

import re
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CaptionFilterResult:
    """Result from caption filtering"""
    decision: str  # 'ACCEPT', 'REJECT', 'PASS'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    matched_keywords: list
    reason: str


class CaptionFilter:
    """
    Pre-filter images based on caption keywords.
    
    Decision Logic:
    - WHITELIST match → ACCEPT (likely circuit)
    - BLACKLIST match → REJECT (not circuit)
    - No match → PASS (let DINOv2 decide)
    - Both matches → BLACKLIST wins (REJECT)
    """
    
    # Strong circuit indicators (ACCEPT)
    WHITELIST_KEYWORDS = [
        # Direct circuit mentions
        r'\bquantum circuit\b',
        r'\bcircuit diagram\b',
        r'\bgate sequence\b',
        r'\bqubit circuit\b',
        r'\bquantum gate\b',
        r'\bquantum pooling layer\b',
        
        # Specific algorithms with circuits
        r'\bgrover circuit\b',
        r'\bshor circuit\b',
        r'\bqaoa circuit\b',
        r'\bvqe circuit\b',
        r'\bbell state circuit\b',
        r'\bentanglement circuit\b',
        r'\bteleportation circuit\b',
        r'\bsuperdense coding circuit\b',
        r'\bqft circuit\b',
        r'\bqpe circuit\b',
        r'\bhhl circuit\b',
        
        # Circuit components
        r'\bqubit wire\b',
        r'\bquantum wire\b',
        r'\bcircuit implementation\b',
        r'\bgate implementation\b',
        r'\bcontrol qubit\b',
        r'\btarget qubit\b',
        r'\bancilla qubit\b',
    ]
    
    # Strong non-circuit indicators (REJECT)
    BLACKLIST_KEYWORDS = [
        # Plots and graphs
        r'\bplot of\b',
        r'\bgraph of\b',
        r'\bbar chart\b',
        r'\bhistogram\b',
        r'\bscatter plot\b',
        r'\bline plot\b',
        r'\bheatmap\b',
        r'\bcontour plot\b',
        
        # Performance metrics
        r'\berror rate\b',
        r'\bfidelity plot\b',
        r'\bprobability distribution\b',
        r'\bresults plot\b',
        r'\bperformance plot\b',
        
        # Non-quantum circuits
        r'\belectrical circuit\b',
        r'\belectronic circuit\b',
        r'\bclassical circuit\b',
        
        # Diagrams (not circuits)
        r'\bflow ?chart\b',
        r'\bblock diagram\b',
        r'\bflow diagram\b',
        r'\barchitecture diagram\b',
        r'\bsystem diagram\b',
        r'\bnetwork topology\b',
        r'\bdata flow\b',
        
        # Setup/Protocol (not circuits)
        r'\bexperimental setup\b',
        r'\bprotocol diagram\b',
        r'\bschematic diagram\b',
        
        # Matrices and mathematical representations
        r'\bunitary matrix\b',
        r'\bdensity matrix\b',
        r'\bstate matrix\b',
        r'\boperator matrix\b',
        r'\bhamiltonian matrix\b',
        r'\btransformation matrix\b',
        r'\bmatrix representation\b',
        r'\bmatrix form\b',
        r'\bmatrix notation\b',

        #More specific Plotting keywords
        r'\blattice\b',
        r'\btensor\b',
        r'\bgauge\b',
        r'\bdynamics\b',
        r'\bcollision\b',
    ]

    
    def __init__(self):
        """Initialize caption filter with compiled regex patterns and statistics tracker."""
        self.whitelist_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.WHITELIST_KEYWORDS
        ]
        self.blacklist_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.BLACKLIST_KEYWORDS
        ]
        
        # Statistics tracking
        self.stats = {
            'whitelist_matches': {},  # keyword -> count
            'blacklist_matches': {},  # keyword -> count
            'total_accept': 0,
            'total_reject': 0,
            'total_pass': 0,
        }
        
        # Initialize counters for each keyword
        for keyword in self.WHITELIST_KEYWORDS:
            self.stats['whitelist_matches'][keyword] = 0
        for keyword in self.BLACKLIST_KEYWORDS:
            self.stats['blacklist_matches'][keyword] = 0
    
    def filter(self, caption: str) -> CaptionFilterResult:
        """
        Filter image based on caption text.
        
        Args:
            caption: Caption text to analyze
            
        Returns:
            CaptionFilterResult with decision (ACCEPT/REJECT/PASS)
        """
        if not caption or not caption.strip():
            return CaptionFilterResult(
                decision='PASS',
                confidence='LOW',
                matched_keywords=[],
                reason='No caption available'
            )
        
        caption_lower = caption.lower()
        
        # Check BLACKLIST first (higher priority)
        blacklist_matches = []
        for i, pattern in enumerate(self.blacklist_patterns):
            match = pattern.search(caption_lower)
            if match:
                matched_keyword = self.BLACKLIST_KEYWORDS[i]
                blacklist_matches.append(match.group(0))
                # Track statistics
                self.stats['blacklist_matches'][matched_keyword] += 1
        
        # Check WHITELIST
        whitelist_matches = []
        for i, pattern in enumerate(self.whitelist_patterns):
            match = pattern.search(caption_lower)
            if match:
                matched_keyword = self.WHITELIST_KEYWORDS[i]
                whitelist_matches.append(match.group(0))
                # Track statistics
                self.stats['whitelist_matches'][matched_keyword] += 1
        
        # Decision logic: BLACKLIST wins if both present
        if blacklist_matches:
            self.stats['total_reject'] += 1
            return CaptionFilterResult(
                decision='REJECT',
                confidence='HIGH',
                matched_keywords=blacklist_matches,
                reason=f'Blacklist match: {blacklist_matches[0]}'
            )
        
        if whitelist_matches:
            self.stats['total_accept'] += 1
            return CaptionFilterResult(
                decision='ACCEPT',
                confidence='HIGH',
                matched_keywords=whitelist_matches,
                reason=f'Whitelist match: {whitelist_matches[0]}'
            )
        
        # No matches - pass to DINOv2
        self.stats['total_pass'] += 1
        return CaptionFilterResult(
            decision='PASS',
            confidence='LOW',
            matched_keywords=[],
            reason='No whitelist/blacklist match - defer to DINOv2'
        )
    
    def get_statistics(self) -> dict:
        """
        Get keyword matching statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()
    
    def print_statistics(self):
        """Print a formatted statistics report."""
        print("\n" + "="*70)
        print("CAPTION FILTER STATISTICS")
        print("="*70)
        
        # Overall summary
        total = self.stats['total_accept'] + self.stats['total_reject'] + self.stats['total_pass']
        print(f"\nTotal Images Processed: {total}")
        print(f" ACCEPTED (whitelist): {self.stats['total_accept']}")
        print(f" REJECTED (blacklist): {self.stats['total_reject']}")
        print(f" PASSED to DINOv2:     {self.stats['total_pass']}")
        
        # Whitelist keywords
        print("\n" + "-"*70)
        print("WHITELIST KEYWORDS (Circuit Indicators)")
        print("-"*70)
        whitelist_sorted = sorted(
            self.stats['whitelist_matches'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for keyword, count in whitelist_sorted:
            if count > 0:
                print(f"  {count:4d}  {keyword}")
        
        if not any(count > 0 for _, count in whitelist_sorted):
            print("  (No whitelist keywords matched)")
        
        # Blacklist keywords
        print("\n" + "-"*70)
        print("BLACKLIST KEYWORDS (Non-Circuit Indicators)")
        print("-"*70)
        blacklist_sorted = sorted(
            self.stats['blacklist_matches'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for keyword, count in blacklist_sorted:
            if count > 0:
                print(f"  {count:4d}  {keyword}")
        
        if not any(count > 0 for _, count in blacklist_sorted):
            print("  (No blacklist keywords matched)")
        
        print("="*70 + "\n")


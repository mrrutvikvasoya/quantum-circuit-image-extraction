"""
Module 4: Detection Orchestrator

Uses DINOv2 embedding-based detection for quantum circuit classification.
Combines semantic similarity (centroid + KNN) for high accuracy.
Includes caption-based pre-filtering for efficiency.
"""

import logging
import torch
from pathlib import Path
from typing import Optional

from PIL import Image

from utils import DetectionResult, DetectionScores, ConfigLoader, PdfData
from detectors.embedding_detector import CircuitDetector
from detectors.embedding_encoder import DINOv2Encoder
from detectors.quantum_problem_classifier import QuantumProblemClassifier
from detectors.caption_filter import CaptionFilter
from detectors.visual_gate_detector import VisualQuantumGateDetector
from detectors import embedding_config as emb_config

logger = logging.getLogger("ProjectNLP.DetectionOrchestrator")


class DetectionOrchestrator:
    """
    Orchestrates quantum circuit detection using DINOv2 embeddings.
    
    Detection method:
    - Caption pre-filter (WHITELIST/BLACKLIST/PASS)
    - Generates 384-dim embeddings using DINOv2
    - Compares to reference centroids (circuit vs non-circuit)
    - Finds K-nearest neighbors in reference set
    - Combines both scores for final decision
    """
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        
        # Auto-detect GPU
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.info("💻 No GPU detected, using CPU")
        
        self.device = device
        
        logger.info("Initializing DINOv2 embedding-based circuit detector...")
        
        # Sync embedding thresholds/weights from config if provided
        emb_cfg = config.get("detection.embedding", {})
        emb_config.COMBINED_THRESHOLD = emb_cfg.get("combinedThreshold", emb_config.COMBINED_THRESHOLD)
        emb_config.CENTROID_WEIGHT = emb_cfg.get("centroidWeight", emb_config.CENTROID_WEIGHT)
        emb_config.KNN_WEIGHT = emb_cfg.get("knnWeight", emb_config.KNN_WEIGHT)
        emb_config.KNN_K = emb_cfg.get("kNeighbors", emb_config.KNN_K)
        emb_config.CONFIDENCE_HIGH = emb_cfg.get("highConfidence", emb_config.CONFIDENCE_HIGH)
        emb_config.CONFIDENCE_MEDIUM = emb_cfg.get("mediumConfidence", emb_config.CONFIDENCE_MEDIUM)
        
        # Initialize encoder and detector
        self.encoder = DINOv2Encoder(device=self.device)
        self.detector = CircuitDetector(self.encoder)
        
        # Load pre-built index
        try:
            self.detector.load_index()
            logger.info("DINOv2 detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load detector index: {e}")
            raise
        
        # Initialize caption filter
        logger.info("Initializing caption pre-filter...")
        self.caption_filter = CaptionFilter()
        logger.info("Caption filter initialized")
        
        # Statistics tracking
        self.stats = {
            'caption_accept': 0,
            'caption_reject': 0,
            'caption_pass': 0,
            'dinov2_processed': 0
        }
        
        # Size constraints
        self.minWidth = config.get("detection.minWidth", 50)
        self.minHeight = config.get("detection.minHeight", 30)
        
        # Initialize visual gate detector
        logger.info("Initializing visual gate detector (EasyOCR)...")
        gate_cfg = config.get("gateDetection", {})
        self.gate_detector = VisualQuantumGateDetector(
            use_gpu=gate_cfg.get("useGpu", False),
            config=config
        )
        logger.info("Visual gate detector initialized")
        
        # Initialize quantum problem classifier
        logger.info("Initializing quantum problem classifier...")
        problem_cfg = config.get("problemClassification", {})
        problem_device = problem_cfg.get("device", self.device)  # Use auto-detected device
        self.problem_classifier = QuantumProblemClassifier(device=problem_device)
        logger.info("Problem classifier initialized")
    
    def get_statistics(self) -> dict:
        """Get caption filter statistics."""
        return self.stats.copy()
    
    def detect(
        self,
        imagePath: str,
        captionText: str = "",
        surroundingText: str = "",
        pdfData: Optional[PdfData] = None,
        figureNumber: Optional[int] = None,
        imageInfo: Optional['ImageInfo'] = None  # NEW: Add imageInfo parameter
    ) -> DetectionResult:
        """
        Detect if an image contains a quantum circuit.
        
        Args:
            imagePath: Path to image file
            captionText: Caption text (optional, for metadata)
            surroundingText: Surrounding text (optional, for metadata)
            pdfData: PDF metadata (optional)
            figureNumber: Figure number (optional)
            imageInfo: Complete ImageInfo object (optional, for referencedText)
        
        Returns:
            DetectionResult with classification and confidence
        """
        imageName = Path(imagePath).name
        logger.debug(f"Processing: {imageName}")
        
        scores = DetectionScores()
        negativeFeatures = []
        
        # Size safety check
        try:
            with Image.open(imagePath) as img:
                w, h = img.size
                if w < self.minWidth or h < self.minHeight:
                    logger.debug(f"Image too small: {w}x{h}")
                    return DetectionResult(
                        isQuantumCircuit=False,
                        confidence=0.0,
                        decisionTier="too_small",
                        scores=scores,
                        gatesFound=[],
                        symbolsFound=[],
                        keywordsFound=[],
                        negativeFeatures=["too_small"]
                    )
        except Exception as e:
            logger.error(f"Error reading image {imagePath}: {e}")
            return DetectionResult(
                isQuantumCircuit=False,
                confidence=0.0,
                decisionTier="error",
                scores=scores,
                gatesFound=[],
                symbolsFound=[],
                keywordsFound=[],
                negativeFeatures=["read_error"]
            )
        
        # === Caption-based pre-filtering ===
        caption_result = self.caption_filter.filter(captionText)
        
        if caption_result.decision == 'ACCEPT':
            # Whitelist match - validate with IMPROVED gate detection
            self.stats['caption_accept'] += 1
            logger.debug(f"  Caption ACCEPT: {caption_result.reason}")
            
            # Run IMPROVED gate detection for validation
            gates_found = []
            gate_confidence_scores = {}
            
            try:
                gate_result = self.gate_detector.detect_gates(imagePath)
                if gate_result and gate_result.get('gates'):
                    gates_found = gate_result['gates']
                    gate_confidence_scores = gate_result.get('confidence_scores', {})
            except Exception as e:
                logger.warning(f"Gate detection failed: {e}")
            
            # VALIDATION: If no gates found, reclassify as non-circuit
            if not gates_found:
                logger.debug(f"  No gates found → Reclassifying as NON-CIRCUIT")
                return DetectionResult(
                    isQuantumCircuit=False,
                    confidence=0.8,
                    decisionTier="caption_whitelist_no_gates",
                    scores=scores,
                    gatesFound=[],
                    symbolsFound=[],
                    keywordsFound=caption_result.matched_keywords,
                    negativeFeatures=["no_gates_found"]
                )
            
            # Gates found - proceed with classification
            problem_category = 'Unknown'
            problem_confidence = 'LOW'
            problem_score = 0.0
            
            try:
                # =====================================================================
                # Problem Classification: Cascading Fallback Strategy
                # Priority: 1) caption+referenced → 2) +surrounding → 3) +abstract+title
                # =====================================================================
                
                # Step 1: Caption + Referenced text (most specific)
                referenced_text = imageInfo.referencedText if imageInfo else ''
                problem_result = self.problem_classifier.classify(
                    caption=captionText,
                    surrounding_text=referenced_text
                )
                problem_category = problem_result.get('category', 'Unknown')
                problem_confidence = problem_result.get('confidence', 'LOW')
                problem_score = problem_result.get('score', 0.0)
                logger.debug(f"  Classification Step 1 (caption+referenced): {problem_category} ({problem_confidence})")
                
                # Step 2: If Unknown, add surrounding text
                if problem_category == 'Unknown' and surroundingText:
                    logger.debug("  Step 1 Unknown → Adding surrounding text...")
                    problem_result = self.problem_classifier.classify(
                        caption=captionText,
                        surrounding_text=' '.join(filter(None, [referenced_text, surroundingText])).strip()
                    )
                    problem_category = problem_result.get('category', 'Unknown')
                    problem_confidence = problem_result.get('confidence', 'LOW')
                    problem_score = problem_result.get('score', 0.0)
                    logger.debug(f"  Classification Step 2 (+ surrounding): {problem_category} ({problem_confidence})")
                
                # Step 3: If still Unknown, add abstract + title
                if problem_category == 'Unknown' and pdfData:
                    logger.debug("  Step 2 Unknown → Adding abstract + title...")
                    problem_result = self.problem_classifier.classify(
                        title=pdfData.title if pdfData.title else '',
                        abstract=pdfData.abstract if pdfData.abstract else '',
                        caption=captionText,
                        surrounding_text=' '.join(filter(None, [referenced_text, surroundingText])).strip()
                    )
                    problem_category = problem_result.get('category', 'Unknown')
                    problem_confidence = problem_result.get('confidence', 'LOW')
                    problem_score = problem_result.get('score', 0.0)
                    logger.debug(f"  Classification Step 3 (+ abstract+title): {problem_category} ({problem_confidence})")
                
            except Exception as e:
                logger.warning(f"Problem classification failed: {e}")
            
            return DetectionResult(
                isQuantumCircuit=True,
                confidence=0.95,
                decisionTier="caption_whitelist",
                scores=scores,
                gatesFound=gates_found,
                symbolsFound=[],
                keywordsFound=caption_result.matched_keywords,
                negativeFeatures=[],
                gateConfidenceScores=gate_confidence_scores,
                problemCategory=problem_category,
                problemConfidence=problem_confidence,
                problemScore=problem_score
            )
        
        elif caption_result.decision == 'REJECT':
            # Blacklist match - reject without DINOv2
            self.stats['caption_reject'] += 1
            logger.debug(f"  Caption REJECT: {caption_result.reason}")
            return DetectionResult(
                isQuantumCircuit=False,
                confidence=0.95,
                decisionTier="caption_blacklist",
                scores=scores,
                gatesFound=[],
                symbolsFound=[],
                keywordsFound=caption_result.matched_keywords,
                negativeFeatures=["caption_blacklist"]
            )
        
        else:
            # PASS - continue to DINOv2
            self.stats['caption_pass'] += 1
            logger.debug(f"  Caption PASS: {caption_result.reason}")
        
        # Run DINOv2 embedding detection
        try:
            result = self.detector.detect(imagePath)
            
            # Store embedding scores
            scores.embedding_centroid = result.centroid_score
            scores.embedding_knn = result.knn_score
            scores.embedding_combined = result.combined_score
            
            combined_score = result.combined_score
            
            # Log detection details
            logger.debug(f"  DINOv2 Combined Score: {combined_score:.3f}")
            logger.debug(f"  Centroid: {result.centroid_score:.3f}, KNN: {result.knn_score:.3f}")
            
            # === TIERED DECISION LOGIC ===
            # HIGH (>=threshold): Require gate validation
            # MEDIUM/LOW (<threshold): Reject
            
            emb_cfg = self.config.get("detection.embedding", {})
            HIGH_THRESHOLD = emb_cfg.get("combinedThreshold", 0.65)
            MEDIUM_THRESHOLD = 0.50
            
            gates_found = []
            gate_confidence_scores = {}
            problem_category = 'Unknown'
            problem_confidence = 'LOW'
            problem_score = 0.0
            
            # --- HIGH SCORE: Validate with IMPROVED gate detection ---
            if combined_score >= HIGH_THRESHOLD:
                logger.debug(f"  HIGH score ({combined_score:.3f} >= {HIGH_THRESHOLD}) → Running IMPROVED gate validation...")
                
                # Run IMPROVED gate detection for validation
                try:
                    gate_result = self.gate_detector.detect_gates(imagePath, verbose=False)
                    if gate_result['success']:
                        gates_found = gate_result.get('gates', [])
                        gate_confidence_scores = gate_result.get('confidence_scores', {})
                except Exception as e:
                    logger.warning(f"  Gate extraction error: {e}")
                
                # VALIDATION: If no gates found, reclassify as non-circuit
                if not gates_found:
                    logger.debug(f"  No gates found → Reclassifying as NON-CIRCUIT")
                    return DetectionResult(
                        isQuantumCircuit=False,
                        confidence=1.0 - combined_score,
                        decisionTier="dinov2_high_no_gates",
                        scores=scores,
                        gatesFound=[],
                        gateConfidenceScores={},
                        problemCategory='Unknown',
                        problemConfidence='LOW',
                        problemScore=0.0,
                        symbolsFound=[],
                        keywordsFound=[],
                        negativeFeatures=["no_gates_found"]
                    )
                
                # Gates found - proceed with classification
                logger.debug(f"  Gates found: {gates_found} → CIRCUIT (validated)")
                try:
                    # =====================================================================
                    # Problem Classification: Cascading Fallback Strategy
                    # Priority: 1) caption+referenced → 2) +surrounding → 3) +abstract+title
                    # =====================================================================
                    
                    # Step 1: Caption + Referenced text (most specific)
                    referenced_text = imageInfo.referencedText if imageInfo else ''
                    problem_result = self.problem_classifier.classify(
                        caption=captionText,
                        surrounding_text=referenced_text
                    )
                    problem_category = problem_result.get('category', 'Unknown')
                    problem_confidence = problem_result.get('confidence', 'LOW')
                    problem_score = problem_result.get('score', 0.0)
                    logger.debug(f"  Classification Step 1 (caption+referenced): {problem_category} ({problem_confidence})")
                    
                    # Step 2: If Unknown, add surrounding text
                    if problem_category == 'Unknown' and surroundingText:
                        logger.debug("  Step 1 Unknown → Adding surrounding text...")
                        problem_result = self.problem_classifier.classify(
                            caption=captionText,
                            surrounding_text=' '.join(filter(None, [referenced_text, surroundingText])).strip()
                        )
                        problem_category = problem_result.get('category', 'Unknown')
                        problem_confidence = problem_result.get('confidence', 'LOW')
                        problem_score = problem_result.get('score', 0.0)
                        logger.debug(f"  Classification Step 2 (+ surrounding): {problem_category} ({problem_confidence})")
                    
                    # Step 3: If still Unknown, add abstract + title
                    if problem_category == 'Unknown' and pdfData:
                        logger.debug("  Step 2 Unknown → Adding abstract + title...")
                        problem_result = self.problem_classifier.classify(
                            title=pdfData.title if pdfData.title else '',
                            abstract=pdfData.abstract if pdfData.abstract else '',
                            caption=captionText,
                            surrounding_text=' '.join(filter(None, [referenced_text, surroundingText])).strip()
                        )
                        problem_category = problem_result.get('category', 'Unknown')
                        problem_confidence = problem_result.get('confidence', 'LOW')
                        problem_score = problem_result.get('score', 0.0)
                        logger.debug(f"  Classification Step 3 (+ abstract+title): {problem_category} ({problem_confidence})")
                    
                except Exception as e:
                    logger.warning(f"  Problem classification error: {e}")
                
                return DetectionResult(
                    isQuantumCircuit=True,
                    confidence=combined_score,
                    decisionTier="dinov2_high",
                    scores=scores,
                    gatesFound=gates_found,
                    gateConfidenceScores=gate_confidence_scores,
                    problemCategory=problem_category,
                    problemConfidence=problem_confidence,
                    problemScore=problem_score,
                    symbolsFound=[],
                    keywordsFound=[],
                    negativeFeatures=[]
                )
            
            # --- MEDIUM or LOW SCORE: Reject ---
            score_tier = "MEDIUM" if combined_score >= MEDIUM_THRESHOLD else "LOW"
            decision_tier = f"dinov2_{score_tier.lower()}_rejected"
            logger.debug(f"  {score_tier} score ({combined_score:.3f}) → NOT CIRCUIT (rejected)")
            
            return DetectionResult(
                isQuantumCircuit=False,
                confidence=1.0 - combined_score,
                decisionTier=decision_tier,
                scores=scores,
                gatesFound=[],
                symbolsFound=[],
                keywordsFound=[],
                negativeFeatures=[f"dinov2_{score_tier.lower()}"]
            )
            
        except Exception as e:
            logger.error(f"Embedding detection error for {imagePath}: {e}")
            return DetectionResult(
                isQuantumCircuit=False,
                confidence=0.0,
                decisionTier="detection_error",
                scores=scores,
                gatesFound=[],
                symbolsFound=[],
                keywordsFound=[],
                negativeFeatures=["detection_error"]
            )

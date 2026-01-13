"""
Quantum Circuit Detector using Embeddings
"""
import os
import pickle
import numpy as np
import faiss
from typing import Dict, Union, List
from PIL import Image
from dataclasses import dataclass
from . import embedding_config as config
from .embedding_encoder import DINOv2Encoder


@dataclass
class DetectionResult:
    """Result of circuit detection"""
    is_circuit: bool
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    combined_score: float
    centroid_score: float
    knn_score: float
    centroid_distance_circuit: float
    centroid_distance_non_circuit: float
    knn_circuit_count: int
    knn_non_circuit_count: int
    knn_total: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'is_circuit': self.is_circuit,
            'confidence': self.confidence,
            'combined_score': round(self.combined_score, 4),
            'centroid_score': round(self.centroid_score, 4),
            'knn_score': round(self.knn_score, 4),
            'centroid_distance_circuit': round(self.centroid_distance_circuit, 4),
            'centroid_distance_non_circuit': round(self.centroid_distance_non_circuit, 4),
            'knn_circuit_count': self.knn_circuit_count,
            'knn_non_circuit_count': self.knn_non_circuit_count,
            'knn_total': self.knn_total
        }


class CircuitDetector:
    """Detect quantum circuits using embedding similarity"""
    
    def __init__(self, encoder: DINOv2Encoder = None):
        """
        Initialize the detector
        
        Args:
            encoder: DINOv2Encoder instance (creates new one if None)
        """
        self.encoder = encoder or DINOv2Encoder()
        self.centroids = None
        self.faiss_index = None
        self.labels = None
        self.loaded = False
    
    def load_index(self):
        """Load the pre-built index from disk"""
        print("\nLoading Reference Index...")
        
        # Load centroids
        with open(config.CENTROIDS_FILE, 'rb') as f:
            self.centroids = pickle.load(f)

        
        # Load FAISS index
        self.faiss_index = faiss.read_index(config.FAISS_INDEX_FILE)

        
        # Load embeddings to get labels
        with open(config.EMBEDDINGS_FILE, 'rb') as f:
            embeddings_db = pickle.load(f)
        
        # Reconstruct labels
        labels = []
        n_circuit = len(embeddings_db['circuit']['embeddings'])
        n_non_circuit = len(embeddings_db['non_circuit']['embeddings'])
        
        labels.extend([1] * n_circuit)  # 1 for circuit
        labels.extend([0] * n_non_circuit)  # 0 for non-circuit
        
        self.labels = np.array(labels)

        
        self.loaded = True
        print("Index loaded successfully!\n")
    
    def detect(self, image: Union[str, Image.Image]) -> DetectionResult:
        """
        Detect if an image contains a quantum circuit
        
        Args:
            image: Path to image or PIL Image
            
        Returns:
            DetectionResult object
        """
        if not self.loaded:
            raise RuntimeError("Index not loaded! Call load_index() first.")
        
        # Generate embedding for the image
        embedding = self.encoder.encode_single(image)
        
        # Method A: Centroid-based detection
        centroid_score, dist_circuit, dist_non_circuit = self._centroid_detection(embedding)
        
        # Method B: KNN-based detection
        knn_score, circuit_count, non_circuit_count = self._knn_detection(embedding)
        
        # Combine scores
        combined_score = (config.CENTROID_WEIGHT * centroid_score + 
                         config.KNN_WEIGHT * knn_score)
        
        # Determine if circuit
        is_circuit = combined_score >= config.COMBINED_THRESHOLD
        
        # Determine confidence
        if combined_score >= config.CONFIDENCE_HIGH:
            confidence = 'HIGH'
        elif combined_score >= config.CONFIDENCE_MEDIUM:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return DetectionResult(
            is_circuit=is_circuit,
            confidence=confidence,
            combined_score=combined_score,
            centroid_score=centroid_score,
            knn_score=knn_score,
            centroid_distance_circuit=dist_circuit,
            centroid_distance_non_circuit=dist_non_circuit,
            knn_circuit_count=circuit_count,
            knn_non_circuit_count=non_circuit_count,
            knn_total=config.KNN_K
        )
    
    def detect_batch(self, images: List[Union[str, Image.Image]]) -> List[DetectionResult]:
        """
        Detect circuits in multiple images
        
        Args:
            images: List of image paths or PIL Images
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results
    
    def _centroid_detection(self, embedding: np.ndarray) -> tuple:
        """
        Method A: Centroid-based detection
        
        Returns:
            (score, distance_to_circuit, distance_to_non_circuit)
        """
        # Compute distances to centroids (L2 distance with normalized vectors = cosine distance)
        dist_circuit = np.linalg.norm(embedding - self.centroids['circuit'])
        dist_non_circuit = np.linalg.norm(embedding - self.centroids['non_circuit'])
        
        # Convert to similarity score
        # Lower distance to circuit centroid = higher score
        # We use the ratio: closer to circuit vs closer to non-circuit
        if dist_circuit + dist_non_circuit == 0:
            score = 0.5
        else:
            # Score based on relative distances
            # If dist_circuit is small and dist_non_circuit is large, score is high
            score = dist_non_circuit / (dist_circuit + dist_non_circuit)
        
        return score, dist_circuit, dist_non_circuit
    
    def _knn_detection(self, embedding: np.ndarray, k: int = None) -> tuple:
        """
        Method B: K-Nearest Neighbor detection
        
        Returns:
            (score, circuit_count, non_circuit_count)
        """
        k = k or config.KNN_K
        
        # Ensure k doesn't exceed total number of vectors
        k = min(k, self.faiss_index.ntotal)
        
        # Search for k nearest neighbors
        embedding_2d = embedding.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(embedding_2d, k)
        
        # Count circuit vs non-circuit neighbors
        neighbor_labels = self.labels[indices[0]]
        circuit_count = np.sum(neighbor_labels == 1)
        non_circuit_count = np.sum(neighbor_labels == 0)
        
        # Score is the ratio of circuit neighbors
        score = circuit_count / k if k > 0 else 0.0
        
        return score, int(circuit_count), int(non_circuit_count)

"""
Configuration for Embedding-Based Quantum Circuit Detection
"""
import os

# Model Configuration
MODEL_NAME = "facebook/dinov2-small"  # 384-dim embeddings
EMBEDDING_DIM = 384
IMAGE_SIZE = 224

# Detection Thresholds
CENTROID_THRESHOLD = 0.70  # Threshold for centroid-based detection
KNN_THRESHOLD = 0.70       # Threshold for KNN-based detection
COMBINED_THRESHOLD = 0.70  # Threshold for combined score
KNN_K = 10                 # Number of nearest neighbors to consider

# Weights for combining scores
CENTROID_WEIGHT = 0.5
KNN_WEIGHT = 0.5

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "embedding_index")

# Index files
EMBEDDINGS_FILE = os.path.join(INDEX_DIR, "embeddings.pkl")
CENTROIDS_FILE = os.path.join(INDEX_DIR, "centroids.pkl")
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")

# Confidence levels
CONFIDENCE_HIGH = 0.80
CONFIDENCE_MEDIUM = 0.60

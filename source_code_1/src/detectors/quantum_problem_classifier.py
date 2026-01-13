"""
Quantum Problem Classifier
Two-stage approach: Explicit phrase matching + SciBERT embedding similarity
"""
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Import phrases and reference texts from config
from detectors.quantum_problem_config import EXPLICIT_PHRASES, REFERENCE_TEXTS


class QuantumProblemClassifier:
    """
    Classify quantum computing papers into problem categories
    Stage 1: Explicit phrase matching (fast, high precision)
    Stage 2: SciBERT embedding similarity (semantic understanding)
    """
    
    # Problem categories
    CATEGORIES = [
        'Grover_Algorithm',
        'Shor_Algorithm',
        'QFT_QPE',  # Quantum Fourier Transform / Phase Estimation
        'VQE',
        'QAOA',
        'Quantum_Simulation',
        'Quantum_Machine_Learning',
        'Quantum_Cryptography',
        'Error_Correction',
        'Hardware_Architecture',
        'Benchmarking',
        'Optimization_Problems',
        'Unknown'
    ]
    
    # Import from config file
    EXPLICIT_PHRASES = EXPLICIT_PHRASES
    REFERENCE_TEXTS = REFERENCE_TEXTS
    
    # Thresholds
    SIMILARITY_THRESHOLD = 0.65  # Minimum similarity to classify
    MIN_MARGIN = 0.10  # Minimum difference between top 2 categories
    
    def __init__(self, model_name: str = 'allenai/scibert_scivocab_uncased', device: str = None):
        """
        Initialize the classifier
        
        Args:
            model_name: SciBERT model name
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading SciBERT model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load SciBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Generate reference embeddings (one-time)
        print("Generating reference embeddings...")
        self.reference_embeddings = self._generate_reference_embeddings()
        print(f"✓ Generated embeddings for {len(self.reference_embeddings)} categories")
    
    def _generate_reference_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for all reference texts"""
        embeddings = {}
        for category, text in self.REFERENCE_TEXTS.items():
            embeddings[category] = self._get_embedding(text)
        return embeddings
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate SciBERT embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def classify_stage1(self, text: str) -> Optional[Dict]:
        """
        Stage 1: Explicit phrase matching
        
        Args:
            text: Combined text (title + abstract + caption + description)
            
        Returns:
            Classification result or None if no match
        """
        text_normalized = self._normalize_text(text)
        
        # Check each category's phrases
        for category, phrases in self.EXPLICIT_PHRASES.items():
            for phrase in phrases:
                if phrase.lower() in text_normalized:
                    return {
                        'category': category,
                        'confidence': 'HIGH',
                        'score': 1.0,
                        'method': 'explicit_match',
                        'matched_phrase': phrase
                    }
        
        return None
    
    def classify_stage2(self, text: str) -> Dict:
        """
        Stage 2: SciBERT embedding similarity
        
        Args:
            text: Combined text
            
        Returns:
            Classification result
        """
        # Generate embedding for input text
        text_embedding = self._get_embedding(text).reshape(1, -1)
        
        # Compute similarities with all reference categories
        similarities = {}
        for category, ref_embedding in self.reference_embeddings.items():
            ref_embedding = ref_embedding.reshape(1, -1)
            sim = cosine_similarity(text_embedding, ref_embedding)[0][0]
            similarities[category] = float(sim)
        
        # Sort by similarity
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top category and score
        top_category, top_score = sorted_sims[0]
        second_score = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
        margin = top_score - second_score
        
        # Decision logic with threshold and margin
        if top_score >= self.SIMILARITY_THRESHOLD and margin >= self.MIN_MARGIN:
            category = top_category
            if top_score >= 0.80:
                confidence = 'HIGH'
            elif top_score >= 0.70:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
        else:
            category = 'Unknown'
            confidence = 'LOW'
            # DEBUG: Log why it's Unknown
            import logging
            logger = logging.getLogger("ProjectNLP.QuantumProblemClassifier")
            if top_score < self.SIMILARITY_THRESHOLD:
                logger.debug(f"Unknown: Top score {top_score:.3f} < threshold {self.SIMILARITY_THRESHOLD}")
            elif margin < self.MIN_MARGIN:
                logger.debug(f"Unknown: Margin {margin:.3f} < min_margin {self.MIN_MARGIN} (top: {top_category}={top_score:.3f}, 2nd: {sorted_sims[1][0]}={second_score:.3f})")

        
        return {
            'category': category,
            'confidence': confidence,
            'score': top_score,
            'margin': margin,
            'method': 'embedding_similarity',
            'top_3_categories': sorted_sims[:3]
        }
    
    def classify(
        self,
        title: str = '',
        abstract: str = '',
        caption: str = '',
        description: str = '',
        surrounding_text: str = ''  # NEW: Add surrounding_text parameter
    ) -> Dict:
        """
        Main classification method
        
        Args:
            title: Paper title
            abstract: Paper abstract
            caption: Image captions
            description: Image descriptions
            surrounding_text: Text surrounding the image (NEW)
            
        Returns:
            Classification result with category and confidence
        """
        # Combine all text (now includes surrounding_text)
        combined_text = ' '.join([
            title or '',
            abstract or '',
            caption or '',
            description or '',
            surrounding_text or ''  # NEW: Include surrounding text
        ]).strip()
        
        # DEBUG: Log input text for troubleshooting
        import logging
        logger = logging.getLogger("ProjectNLP.QuantumProblemClassifier")
        logger.debug(f"Classification input - Title: {len(title)} chars, Abstract: {len(abstract)} chars, "
                    f"Caption: {len(caption)} chars, Surrounding: {len(surrounding_text)} chars, "
                    f"Total: {len(combined_text)} chars")
        
        if not combined_text:
            logger.warning("No text provided for classification")
            return {
                'category': 'Unknown',
                'confidence': 'LOW',
                'score': 0.0,
                'method': 'no_text',
                'error': 'No text provided'
            }
        
        # Stage 1: Try explicit phrase matching
        result = self.classify_stage1(combined_text)
        if result:
            return result
        
        # Stage 2: Use SciBERT similarity
        result = self.classify_stage2(combined_text)
        return result

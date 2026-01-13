"""
DINOv2 Encoder for generating image embeddings
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import Union, List
from . import embedding_config as config


class DINOv2Encoder:
    """Encoder class for generating embeddings using DINOv2"""
    
    def __init__(self, model_name: str = config.MODEL_NAME, device: str = None):
        """
        Initialize the DINOv2 encoder
        
        Args:
            model_name: HuggingFace model name
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading DINOv2 model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> Image.Image:
        """
        Preprocess image for encoding
        
        Args:
            image: Path to image or PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path or PIL Image")
        
        # Resize to model input size
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.LANCZOS)
        return image
    
    def encode_single(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Generate embedding for a single image
        
        Args:
            image: Path to image or PIL Image
            
        Returns:
            Normalized embedding vector (384-dim)
        """
        # Preprocess
        image = self.preprocess_image(image)
        
        # Process with model
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # L2 normalize
        embedding = self._normalize(embedding)
        
        return embedding
    
    def encode_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple images
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            
        Returns:
            Array of normalized embeddings (N x 384)
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            processed_images = [self.preprocess_image(img) for img in batch]
            
            # Process with model
            inputs = self.processor(images=processed_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(embeddings)
        
        # L2 normalize
        embeddings = self._normalize_batch(embeddings)
        
        return embeddings
    
    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """L2 normalize a single embedding"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @staticmethod
    def _normalize_batch(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize a batch of embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

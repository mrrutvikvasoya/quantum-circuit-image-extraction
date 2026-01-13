#!/usr/bin/env python3
"""
Build Reference Index for DINOv2-based Circuit Detection

This script:
1. Loads labeled circuit and non-circuit images
2. Generates DINOv2 embeddings for each image
3. Builds FAISS index for fast similarity search
4. Computes centroids for circuit and non-circuit classes
5. Saves everything to disk

Usage:
    python scripts/build_reference_index.py

Directory structure expected:
    data/labeled_images/
        circuits/        # Folder with circuit images
        non_circuits/    # Folder with non-circuit images
"""

import os
import sys
import pickle
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from detectors.embedding_encoder import DINOv2Encoder
from detectors import embedding_config as config


def load_labeled_images(base_dir: str):
    """
    Load labeled images from directory structure:
    base_dir/
        circuits/
        non_circuits/
    
    Returns:
        dict with 'circuit' and 'non_circuit' keys, each containing list of image paths
    """
    base_path = Path(base_dir)
    
    circuit_dir = base_path / 'circuits'
    non_circuit_dir = base_path / 'non_circuits'
    
    if not circuit_dir.exists():
        raise ValueError(f"Circuit directory not found: {circuit_dir}")
    if not non_circuit_dir.exists():
        raise ValueError(f"Non-circuit directory not found: {non_circuit_dir}")
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    circuit_images = []
    for ext in image_extensions:
        circuit_images.extend(circuit_dir.glob(f'*{ext}'))
    
    non_circuit_images = []
    for ext in image_extensions:
        non_circuit_images.extend(non_circuit_dir.glob(f'*{ext}'))
    
    print(f"Found {len(circuit_images)} circuit images")
    print(f"Found {len(non_circuit_images)} non-circuit images")
    
    return {
        'circuit': sorted([str(p) for p in circuit_images]),
        'non_circuit': sorted([str(p) for p in non_circuit_images])
    }


def generate_embeddings(encoder: DINOv2Encoder, image_paths: dict):
    """
    Generate DINOv2 embeddings for all images
    
    Args:
        encoder: DINOv2Encoder instance
        image_paths: Dict with 'circuit' and 'non_circuit' keys
        
    Returns:
        dict with embeddings for each class
    """
    embeddings = {
        'circuit': {
            'paths': [],
            'embeddings': []
        },
        'non_circuit': {
            'paths': [],
            'embeddings': []
        }
    }
    
    # Process circuit images
    print("\nGenerating embeddings for circuit images...")
    for img_path in tqdm(image_paths['circuit']):
        try:
            embedding = encoder.encode_single(img_path)
            embeddings['circuit']['paths'].append(img_path)
            embeddings['circuit']['embeddings'].append(embedding)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Process non-circuit images
    print("\nGenerating embeddings for non-circuit images...")
    for img_path in tqdm(image_paths['non_circuit']):
        try:
            embedding = encoder.encode_single(img_path)
            embeddings['non_circuit']['paths'].append(img_path)
            embeddings['non_circuit']['embeddings'].append(embedding)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Convert to numpy arrays
    embeddings['circuit']['embeddings'] = np.array(embeddings['circuit']['embeddings'])
    embeddings['non_circuit']['embeddings'] = np.array(embeddings['non_circuit']['embeddings'])
    
    print(f"\nGenerated {len(embeddings['circuit']['embeddings'])} circuit embeddings")
    print(f"Generated {len(embeddings['non_circuit']['embeddings'])} non-circuit embeddings")
    
    return embeddings


def compute_centroids(embeddings: dict):
    """
    Compute centroids for circuit and non-circuit classes
    
    Args:
        embeddings: Dict with 'circuit' and 'non_circuit' embeddings
        
    Returns:
        dict with 'circuit' and 'non_circuit' centroids
    """
    centroids = {
        'circuit': np.mean(embeddings['circuit']['embeddings'], axis=0),
        'non_circuit': np.mean(embeddings['non_circuit']['embeddings'], axis=0)
    }
    
    print(f"\nComputed circuit centroid: shape {centroids['circuit'].shape}")
    print(f"Computed non-circuit centroid: shape {centroids['non_circuit'].shape}")
    
    return centroids


def build_faiss_index(embeddings: dict):
    """
    Build FAISS index for fast similarity search
    
    Args:
        embeddings: Dict with 'circuit' and 'non_circuit' embeddings
        
    Returns:
        faiss.Index object
    """
    # Combine all embeddings
    all_embeddings = np.vstack([
        embeddings['circuit']['embeddings'],
        embeddings['non_circuit']['embeddings']
    ])
    
    # Ensure float32 for FAISS
    all_embeddings = all_embeddings.astype('float32')
    
    # Get dimension
    dim = all_embeddings.shape[1]
    
    # Build flat L2 index (exact search)
    index = faiss.IndexFlatL2(dim)
    
    # Add vectors
    index.add(all_embeddings)
    
    print(f"\nBuilt FAISS index:")
    print(f"  Dimension: {dim}")
    print(f"  Total vectors: {index.ntotal}")
    
    return index


def save_index(embeddings: dict, centroids: dict, faiss_index, output_dir: str):
    """
    Save embeddings, centroids, and FAISS index to disk
    
    Args:
        embeddings: Dict with embeddings
        centroids: Dict with centroids
        faiss_index: FAISS index
        output_dir: Directory to save to
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings (contains paths and embeddings)
    embeddings_file = output_path / 'reference_embeddings.pkl'
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"\n✓ Saved embeddings to {embeddings_file}")
    
    # Save centroids
    centroids_file = output_path / 'cluster_centroids.pkl'
    with open(centroids_file, 'wb') as f:
        pickle.dump(centroids, f)
    print(f"✓ Saved centroids to {centroids_file}")
    
    # Save FAISS index
    index_file = output_path / 'reference_index.faiss'
    faiss.write_index(faiss_index, str(index_file))
    print(f"✓ Saved FAISS index to {index_file}")
    
    # Save metadata
    metadata = {
        'n_circuits': len(embeddings['circuit']['embeddings']),
        'n_non_circuits': len(embeddings['non_circuit']['embeddings']),
        'total': len(embeddings['circuit']['embeddings']) + len(embeddings['non_circuit']['embeddings']),
        'embedding_dim': embeddings['circuit']['embeddings'].shape[1]
    }
    
    metadata_file = output_path / 'index_metadata.pkl'
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata to {metadata_file}")
    
    print(f"\nAll files saved to: {output_path}")


def main():
    """Main function to build reference index"""
    
    print("=" * 70)
    print("Building DINOv2 Reference Index")
    print("=" * 70)
    
    # Configuration
    labeled_images_dir = "data/labeled_images"
    output_dir = "models/dinov2_index"
    
    print(f"\nConfiguration:")
    print(f"  Input directory: {labeled_images_dir}")
    print(f"  Output directory: {output_dir}")
    
    # Step 1: Load labeled images
    print("\n" + "=" * 70)
    print("Step 1: Loading labeled images")
    print("=" * 70)
    image_paths = load_labeled_images(labeled_images_dir)
    
    if len(image_paths['circuit']) == 0 or len(image_paths['non_circuit']) == 0:
        print("\nERROR: Need both circuit and non-circuit images!")
        print(f"Circuits: {len(image_paths['circuit'])}")
        print(f"Non-circuits: {len(image_paths['non_circuit'])}")
        sys.exit(1)
    
    # Step 2: Initialize encoder
    print("\n" + "=" * 70)
    print("Step 2: Initializing DINOv2 encoder")
    print("=" * 70)
    encoder = DINOv2Encoder()
    
    # Step 3: Generate embeddings
    print("\n" + "=" * 70)
    print("Step 3: Generating embeddings")
    print("=" * 70)
    embeddings = generate_embeddings(encoder, image_paths)
    
    # Step 4: Compute centroids
    print("\n" + "=" * 70)
    print("Step 4: Computing centroids")
    print("=" * 70)
    centroids = compute_centroids(embeddings)
    
    # Step 5: Build FAISS index
    print("\n" + "=" * 70)
    print("Step 5: Building FAISS index")
    print("=" * 70)
    faiss_index = build_faiss_index(embeddings)
    
    # Step 6: Save everything
    print("\n" + "=" * 70)
    print("Step 6: Saving index")
    print("=" * 70)
    save_index(embeddings, centroids, faiss_index, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Processed {len(embeddings['circuit']['embeddings'])} circuit images")
    print(f"✓ Processed {len(embeddings['non_circuit']['embeddings'])} non-circuit images")
    print(f"✓ Total embeddings: {faiss_index.ntotal}")
    print(f"✓ Embedding dimension: {embeddings['circuit']['embeddings'].shape[1]}")
    print(f"\nReference index built successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

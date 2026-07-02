import os
import faiss
import numpy as np
import logging
import time
from typing import Tuple, Optional
from app.config import FAISS_INDEX_PATH, DEFAULT_TOP_N

# Setup Logging
logger = logging.getLogger(__name__)

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes a 2D numpy array of embeddings to unit length (L2 norm).
    This is required for inner product matching to yield exact cosine similarities.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Prevent division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms

def build_faiss_index(embeddings: np.ndarray, save_path: str = FAISS_INDEX_PATH) -> faiss.IndexFlatIP:
    """
    Builds a FAISS Flat Inner Product index, adds L2-normalized candidate embeddings,
    and persists the index to disk.
    """
    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2D numpy array.")

    dimension = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP with dimension: {dimension} and {len(embeddings)} items...")
    
    start_time = time.time()
    try:
        # Normalize embeddings for Cosine Similarity inside IndexFlatIP
        normalized_embeddings = normalize_vectors(embeddings.astype('float32'))
        
        # Initialize IndexFlatIP
        index = faiss.IndexFlatIP(dimension)
        index.add(normalized_embeddings)
        
        # Save to disk
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(index, save_path)
        logger.info(f"Successfully built and saved FAISS index to '{save_path}' in {time.time() - start_time:.4f}s")
        return index
    except Exception as e:
        logger.error(f"Failed to build or save FAISS index: {e}")
        raise RuntimeError("FAISS indexing failed.") from e

def load_faiss_index(load_path: str = FAISS_INDEX_PATH) -> Optional[faiss.IndexFlatIP]:
    """
    Loads a FAISS index from disk. Returns None if file does not exist.
    """
    if not os.path.exists(load_path):
        logger.info(f"FAISS index file not found at '{load_path}'. Ready for creation.")
        return None
        
    logger.info(f"Loading FAISS index from '{load_path}'...")
    start_time = time.time()
    try:
        index = faiss.read_index(load_path)
        # Ensure it's a Flat Inner Product index
        if not isinstance(index, faiss.IndexFlatIP):
            # Attempting to load as FlatIP - note that read_index determines the index class automatically
            pass
        logger.info(f"FAISS index loaded successfully in {time.time() - start_time:.4f}s")
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None

def search_faiss_index(
    query_embedding: np.ndarray, 
    index: faiss.IndexFlatIP, 
    top_n: int = DEFAULT_TOP_N
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Searches the FAISS index with a single query embedding.
    Returns:
        distances: Cosine similarity scores (1D numpy array of shape [top_n])
        indices: Candidate indices matching original order (1D numpy array of shape [top_n])
    """
    # Ensure query embedding is 2D float32 [1, dimension]
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    query_embedding = query_embedding.astype('float32')
    
    # Normalize query for cosine similarity
    normalized_query = normalize_vectors(query_embedding)
    
    # Cap top_n to total index size to prevent indexing errors
    total_elements = index.ntotal
    actual_top_n = min(top_n, total_elements)
    
    if actual_top_n <= 0:
        logger.warning("FAISS Index is empty. Zero matches returned.")
        return np.array([]), np.array([])
        
    logger.info(f"Executing FAISS search for top-{actual_top_n} candidates...")
    start_time = time.time()
    try:
        # Search returns distances (similarities) and indexes
        distances, indices = index.search(normalized_query, actual_top_n)
        logger.info(f"FAISS search completed in {time.time() - start_time:.4f}s")
        
        # Flatten outputs for easy 1D consumption
        return distances[0], indices[0]
    except Exception as e:
        logger.error(f"FAISS index search failed: {e}")
        raise RuntimeError("FAISS search execution error.") from e

if __name__ == "__main__":
    # Quick Test Execution
    np.random.seed(42)
    test_dim = 128
    num_cands = 50
    mock_embeddings = np.random.randn(num_cands, test_dim).astype('float32')
    
    test_index_path = os.path.join("models", "test_faiss.bin")
    
    # Build Index
    idx = build_faiss_index(mock_embeddings, save_path=test_index_path)
    
    # Reload Index
    reloaded_idx = load_faiss_index(test_index_path)
    
    # Search
    mock_query = np.random.randn(1, test_dim).astype('float32')
    sims, indices = search_faiss_index(mock_query, reloaded_idx, top_n=5)
    
    print("\n--- FAISS Search Test ---")
    print("Top Match Indices:", indices)
    print("Cosine Similarity Scores:", sims)
    
    # Clean up test index
    if os.path.exists(test_index_path):
        os.remove(test_index_path)

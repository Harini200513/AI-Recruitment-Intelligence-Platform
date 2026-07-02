import os
import pickle
import hashlib
import logging
import time
from typing import List, Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    PRIMARY_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    BGE_QUERY_PREFIX,
    EMBEDDINGS_CACHE
)

# Setup Logging
logger = logging.getLogger(__name__)

class EmbeddingManager:
    _instance = None
    _model = None
    _model_name = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # We perform lazy initialization of models
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_loaded = False
        self._load_cache()

    def _load_cache(self) -> None:
        """Loads the local pickle embedding cache from disk if it exists."""
        if self.cache_loaded:
            return
            
        if os.path.exists(EMBEDDINGS_CACHE):
            try:
                start_time = time.time()
                with open(EMBEDDINGS_CACHE, "rb") as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings from disk in {time.time() - start_time:.4f}s")
            except Exception as e:
                logger.warning(f"Could not load embedding cache: {e}. Starting with an empty cache.")
                self.cache = {}
        else:
            logger.info("No embedding cache found on disk. Initializing empty cache.")
            self.cache = {}
        self.cache_loaded = True

    def _save_cache(self) -> None:
        """Persists the updated embedding cache back to disk."""
        try:
            start_time = time.time()
            with open(EMBEDDINGS_CACHE, "wb") as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} cached embeddings to disk in {time.time() - start_time:.4f}s")
        except Exception as e:
            logger.error(f"Error persisting embedding cache to disk: {e}")

    def load_model(self, model_name: str = PRIMARY_EMBEDDING_MODEL) -> SentenceTransformer:
        """
        Loads the SentenceTransformer model using a thread-safe singleton pattern.
        Falls back to the alternative model if loading fails.
        """
        if self._model is not None and self._model_name == model_name:
            return self._model

        logger.info(f"Attempting to load embedding model: {model_name}...")
        try:
            start_time = time.time()
            # Attempt loading primary
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
            logger.info(f"Successfully loaded model '{model_name}' in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to load primary model '{model_name}': {e}.")
            if model_name != FALLBACK_EMBEDDING_MODEL:
                logger.info(f"Attempting fallback to model: {FALLBACK_EMBEDDING_MODEL}...")
                try:
                    start_time = time.time()
                    self._model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
                    self._model_name = FALLBACK_EMBEDDING_MODEL
                    logger.info(f"Successfully loaded fallback model '{FALLBACK_EMBEDDING_MODEL}' in {time.time() - start_time:.2f}s")
                except Exception as fallback_err:
                    logger.critical(f"Critical Error: Fallback model loading failed: {fallback_err}")
                    raise RuntimeError("Could not load any sentence embedding model.") from fallback_err
            else:
                logger.critical(f"Critical Error: Failed loading model: {model_name}")
                raise RuntimeError("Could not load sentence embedding model.") from e
                
        return self._model

    def _get_hash(self, text: str) -> str:
        """Generates a unique SHA-256 hash for a given text string for caching keys."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_embeddings(self, texts: List[str], model_name: str = PRIMARY_EMBEDDING_MODEL, is_query: bool = False) -> np.ndarray:
        """
        Generates embeddings for a list of texts, checking cache entries first.
        Saves new entries back to disk.
        """
        if not texts:
            return np.empty((0, 0))

        # Assure model is loaded
        model = self.load_model(model_name)

        # Prepend query instruction for BGE model if it is a search query
        processed_texts = []
        is_bge = "bge" in model_name.lower() or "bge" in self._model_name.lower()
        
        for text in texts:
            if is_query and is_bge:
                processed_texts.append(f"{BGE_QUERY_PREFIX}{text}")
            else:
                processed_texts.append(text)

        # Identify which items are already cached
        embeddings = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # We cache based on the model name and text hash
        for idx, text in enumerate(processed_texts):
            cache_key = f"{self._model_name}_{self._get_hash(text)}"
            if cache_key in self.cache:
                embeddings[idx] = self.cache[cache_key]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        # Encode uncached items
        if uncached_texts:
            logger.info(f"Encoding {len(uncached_texts)} uncached texts using '{self._model_name}'...")
            start_time = time.time()
            try:
                new_embeddings = model.encode(uncached_texts, show_progress_bar=False, convert_to_numpy=True)
                logger.info(f"Generated {len(uncached_texts)} embeddings in {time.time() - start_time:.4f}s")
                
                # Store back in cache & output list
                for local_idx, global_idx in enumerate(uncached_indices):
                    text_ref = uncached_texts[local_idx]
                    cache_key = f"{self._model_name}_{self._get_hash(text_ref)}"
                    self.cache[cache_key] = new_embeddings[local_idx]
                    embeddings[global_idx] = new_embeddings[local_idx]
                
                # Save cache update
                self._save_cache()
            except Exception as e:
                logger.error(f"Error during embedding generation: {e}")
                raise e
        else:
            logger.info(f"All {len(texts)} embeddings loaded from cache.")

        return np.vstack(embeddings)

    def clear_cache(self) -> None:
        """Wipes the local cache files and resets memory state."""
        self.cache = {}
        if os.path.exists(EMBEDDINGS_CACHE):
            try:
                os.remove(EMBEDDINGS_CACHE)
                logger.info("Embedding cache database file deleted.")
            except Exception as e:
                logger.warning(f"Could not delete embedding cache file: {e}")
        self.cache_loaded = True

if __name__ == "__main__":
    # Test execution
    manager = EmbeddingManager()
    
    test_queries = ["Looking for a Senior Python Developer with Kubernetes skills."]
    test_docs = [
        "Skills: python, SQL, Kubernetes. 5 years experience.",
        "Skills: React, HTML, CSS. Junior developer."
    ]
    
    print("--- Embedding Query ---")
    query_emb = manager.get_embeddings(test_queries, is_query=True)
    print("Query Embedding Shape:", query_emb.shape)
    
    print("\n--- Embedding Documents (Cold Run) ---")
    doc_embs_cold = manager.get_embeddings(test_docs)
    print("Docs Shape:", doc_embs_cold.shape)
    
    print("\n--- Embedding Documents (Warm Run - Cache check) ---")
    doc_embs_warm = manager.get_embeddings(test_docs)
    print("Docs Shape:", doc_embs_warm.shape)

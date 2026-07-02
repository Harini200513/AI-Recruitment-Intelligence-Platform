import logging
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import CrossEncoder

from app.config import CROSS_ENCODER_MODEL

# Setup Logging
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    _instance = None
    _model = None
    _model_name = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CrossEncoderReranker, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def load_model(self, model_name: str = CROSS_ENCODER_MODEL) -> CrossEncoder:
        """
        Loads the CrossEncoder model using a singleton pattern.
        """
        if self._model is not None and self._model_name == model_name:
            return self._model

        logger.info(f"Loading Cross-Encoder model: {model_name}...")
        start_time = time.time()
        try:
            self._model = CrossEncoder(model_name)
            self._model_name = model_name
            logger.info(f"Successfully loaded Cross-Encoder '{model_name}' in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load Cross-Encoder model '{model_name}': {e}.")
            # We raise a runtime error, which will trigger our graceful fallback in the scoring/reranking pipeline.
            raise RuntimeError(f"Could not load Cross-Encoder model: {e}") from e
            
        return self._model

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Applies a sigmoid activation function to normalize logits to [0, 1] range."""
        return 1 / (1 + np.exp(-x))

    def compute_scores(self, query: str, documents: List[str], model_name: str = CROSS_ENCODER_MODEL) -> List[float]:
        """
        Computes relevance scores for (query, document) pairs.
        Applies sigmoid to normalize scores.
        """
        if not documents:
            return []

        try:
            model = self.load_model(model_name)
        except Exception as err:
            logger.warning(f"Reranker failed to load model. Degrading semantic scores gracefully: {err}")
            # Returning empty list to signal the calling module to fall back to retrieval scores
            return []

        logger.info(f"Rerank computation: scoring {len(documents)} document pairs...")
        start_time = time.time()
        try:
            # Construct pairs: [(query, doc_1), (query, doc_2), ...]
            pairs = [[query, doc] for doc in documents]
            
            # Predict raw logits
            raw_scores = model.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
            
            # Normalize raw scores using sigmoid function
            normalized_scores = self.sigmoid(raw_scores)
            
            # Convert to standard Python float list
            scores_list = normalized_scores.tolist()
            logger.info(f"Rerank scoring finished in {time.time() - start_time:.4f}s")
            return scores_list
        except Exception as e:
            logger.error(f"Error during Cross-Encoder prediction: {e}. Falling back to standard FAISS similarities.")
            return []

    def rerank(
        self, 
        query: str, 
        candidate_ids: List[Any], 
        candidate_docs: List[str], 
        retrieval_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidates. If reranker is offline, falls back to FAISS retrieval scores.
        Returns:
            A list of dicts containing candidate_id, rerank_score, and rank index.
        """
        if not candidate_ids or not candidate_docs:
            return []

        # Predict normalized scores
        ce_scores = self.compute_scores(query, candidate_docs)

        reranked_results = []
        is_fallback = len(ce_scores) == 0

        for i, candidate_id in enumerate(candidate_ids):
            # Fallback to normalized FAISS score if CE is offline
            sem_score = retrieval_scores[i] if is_fallback else ce_scores[i]
            
            reranked_results.append({
                "Candidate_ID": candidate_id,
                "Semantic_Score": float(sem_score),
                "Reranked_By_CE": not is_fallback
            })

        # Sort candidates descending by semantic score
        reranked_results.sort(key=lambda x: x["Semantic_Score"], reverse=True)
        
        # Inject new Rank index (1-based)
        for rank_idx, item in enumerate(reranked_results):
            item["Semantic_Rank"] = rank_idx + 1

        logger.info(f"Candidates sorted semantic order. Fallback active: {is_fallback}")
        return reranked_results

if __name__ == "__main__":
    # Test execution
    reranker = CrossEncoderReranker()
    
    test_query = "Python developer with experience in machine learning pipelines."
    test_docs = [
        "Skills: Python, SQL, Git. Experience: 4 years. Built ML models and NLP pipelines.",
        "Skills: React, HTML, CSS. Experience: 2 years. Frontend developer creating dashboard widgets.",
        "Skills: Python, PyTorch. Experience: 1 year. Junior machine learning enthusiast."
    ]
    test_ids = ["C1", "C2", "C3"]
    mock_ret_scores = [0.85, 0.40, 0.70]
    
    print("--- Test Cross-Encoder scoring ---")
    results = reranker.rerank(test_query, test_ids, test_docs, mock_ret_scores)
    for res in results:
        print(f"Rank {res['Semantic_Rank']}: Candidate {res['Candidate_ID']} | Semantic Score: {res['Semantic_Score']:.4f} (Reranked: {res['Reranked_By_CE']})")

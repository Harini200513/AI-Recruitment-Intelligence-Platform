import os
from typing import Dict, Any

# ==========================================
# Paths & Directory Configurations
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure folders exist
for folder in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

CANDIDATES_CSV = os.path.join(DATA_DIR, "sample_candidates.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ranked_candidates.csv")
EMBEDDINGS_CACHE = os.path.join(MODELS_DIR, "embeddings_cache.pkl")
FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "faiss_index.bin")

# ==========================================
# AI Models & Retrieval Settings
# ==========================================
# Sentence Transformer model choices (BGE-Large or MPNet)
PRIMARY_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
FALLBACK_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models perform best with a specific query prefix
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Cross-Encoder for contextual reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Default number of candidates retrieved by FAISS before reranking
DEFAULT_TOP_N = 10

# ==========================================
# Dynamic Hiring Modes & Weights
# ==========================================
# Component weights must sum to 1.0 for each mode
HIRING_MODES: Dict[str, Dict[str, float]] = {
    "Startup Hiring": {
        "semantic": 0.20,
        "skills": 0.30,
        "experience": 0.15,
        "education": 0.05,
        "behavior": 0.05,
        "platform": 0.25,
    },
    "Enterprise Hiring": {
        "semantic": 0.20,
        "skills": 0.15,
        "experience": 0.30,
        "education": 0.20,
        "behavior": 0.10,
        "platform": 0.05,
    },
    "Product Company": {
        "semantic": 0.25,
        "skills": 0.30,
        "experience": 0.20,
        "education": 0.05,
        "behavior": 0.05,
        "platform": 0.15,
    },
    "AI Research": {
        "semantic": 0.30,
        "skills": 0.15,
        "experience": 0.10,
        "education": 0.35,
        "behavior": 0.05,
        "platform": 0.05,
    },
    "Fast Hiring": {
        "semantic": 0.40,
        "skills": 0.35,
        "experience": 0.15,
        "education": 0.05,
        "behavior": 0.03,
        "platform": 0.02,
    }
}

# ==========================================
# Decision Thresholds & Education
# ==========================================
STRONG_HIRE_THRESHOLD = 0.85
GOOD_FIT_THRESHOLD = 0.75
CONSIDER_THRESHOLD = 0.60
NEEDS_IMPROVEMENT_THRESHOLD = 0.45

# Map education levels to numerical values
EDUCATION_MAPPING: Dict[str, float] = {
    "phd": 1.0,
    "doctorate": 1.0,
    "master": 0.85,
    "msc": 0.85,
    "ms": 0.85,
    "mba": 0.80,
    "bachelor": 0.60,
    "bsc": 0.60,
    "bs": 0.60,
    "btech": 0.60,
    "be": 0.60,
    "associate": 0.40,
    "diploma": 0.30,
    "high school": 0.15,
    "none": 0.0
}

# ==========================================
# Skill Learning Estimates & Difficulty
# ==========================================
# Used by the Skill Gap Analysis engine to calculate upskilling requirements
SKILL_METADATA: Dict[str, Dict[str, Any]] = {
    "python": {"difficulty": "Easy", "time_hours": 30},
    "sql": {"difficulty": "Easy", "time_hours": 20},
    "git": {"difficulty": "Easy", "time_hours": 15},
    "docker": {"difficulty": "Medium", "time_hours": 40},
    "kubernetes": {"difficulty": "Hard", "time_hours": 100},
    "aws": {"difficulty": "Medium", "time_hours": 60},
    "gcp": {"difficulty": "Medium", "time_hours": 60},
    "azure": {"difficulty": "Medium", "time_hours": 60},
    "pytorch": {"difficulty": "Hard", "time_hours": 90},
    "tensorflow": {"difficulty": "Hard", "time_hours": 95},
    "keras": {"difficulty": "Medium", "time_hours": 30},
    "pandas": {"difficulty": "Easy", "time_hours": 25},
    "numpy": {"difficulty": "Easy", "time_hours": 20},
    "scikit-learn": {"difficulty": "Medium", "time_hours": 50},
    "fastapi": {"difficulty": "Medium", "time_hours": 30},
    "flask": {"difficulty": "Easy", "time_hours": 20},
    "django": {"difficulty": "Medium", "time_hours": 50},
    "react": {"difficulty": "Medium", "time_hours": 60},
    "javascript": {"difficulty": "Easy", "time_hours": 30},
    "typescript": {"difficulty": "Medium", "time_hours": 40},
    "html": {"difficulty": "Easy", "time_hours": 10},
    "css": {"difficulty": "Easy", "time_hours": 15},
    "spark": {"difficulty": "Hard", "time_hours": 80},
    "hadoop": {"difficulty": "Hard", "time_hours": 70},
    "tableau": {"difficulty": "Easy", "time_hours": 20},
    "powerbi": {"difficulty": "Easy", "time_hours": 25},
    "nlp": {"difficulty": "Hard", "time_hours": 80},
    "computer vision": {"difficulty": "Hard", "time_hours": 90},
    "transformers": {"difficulty": "Hard", "time_hours": 100},
    "llms": {"difficulty": "Hard", "time_hours": 80},
    "mlops": {"difficulty": "Hard", "time_hours": 90},
}

DEFAULT_SKILL_METADATA = {"difficulty": "Medium", "time_hours": 50}

# ==========================================
# Bias Awareness Protected Fields
# ==========================================
# Fields that must be ignored or scrubbed to prevent rating bias
PROTECTED_ATTRIBUTES = ["Candidate_Name", "Gender", "Age", "Religion", "Nationality"]

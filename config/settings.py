"""
Configuration settings for the quantum circuit image extractor.
All tunable parameters are centralized here.
"""

import os

# ================ FILE PATHS ================
ID_FILE = "paper_list_36.txt"
OUTPUT_DIR = "images_test_50_preproc_cached_"
CACHE_DIR = "arxiv_cache"  # Tar.gz source cache
PDF_CACHE_DIR = "arxiv_pdf_cache"  # Extracted PDF cache (faster reaccess)

# ================ EXTRACTION LIMITS ================
MAX_IMAGES = 250
REQUEST_DELAY = 15  # seconds
TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5

# ================ SIMILARITY THRESHOLDS ================
SIMILARITY_THRESHOLD = 0.1  # TF-IDF gate (first filter)
SBERT_MIN_SIM = 0.2  # SBERT gate (optional, see USE_COMBINED_SCORE)

# Weighted combination scoring (recommended)
USE_COMBINED_SCORE = True  # If True, uses weighted combo; if False, uses cascade gates
TFIDF_WEIGHT = 0.5  # Weight for TF-IDF score (lexical precision) - increased for sparse SBERT
SBERT_WEIGHT = 0.5  # Weight for SBERT score (semantic understanding) - lowered for sparse scores
COMBINED_THRESHOLD = 0.4  # Combined score threshold for selection - lowered due to SBERT sparsity

USE_NEGATIVE_PENALTY = True
NEGATIVE_PENALTY_ALPHA = 0.5

# ================ TEXT PROCESSING ================
USE_STEMMING = True
USE_STOPWORDS = True
NORMALIZE_HYPHENS = True

# ================ IMAGE SETTINGS ================
SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

# ================ DEBUGGING ================
ENABLE_DEBUG_PRINTS = True
SAVE_INTERMEDIATE_RESULTS = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# ================ AUTOMATIC SETUP ================
def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PDF_CACHE_DIR, exist_ok=True)
    
    # Create logs directory if logging is enabled
    if SAVE_INTERMEDIATE_RESULTS:
        os.makedirs("logs", exist_ok=True)

# Call setup on import
setup_directories()
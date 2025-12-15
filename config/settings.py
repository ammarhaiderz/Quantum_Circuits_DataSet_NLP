"""
Configuration settings for the quantum circuit image extractor.
All tunable parameters are centralized here.
"""

import os

# ================ FILE PATHS ================
ID_FILE = "paper_list_36.txt"
OUTPUT_DIR = "images_test_50_preproc_cached_"
CACHE_DIR = "arxiv_cache"
PDF_CACHE_DIR = "arxiv_pdf_cache"

# ================ EXTRACTION LIMITS ================
MAX_IMAGES = 350
REQUEST_DELAY = 6
TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5

# ================ SIMILARITY THRESHOLDS ================
SIMILARITY_THRESHOLD = 0.2

# FIXED: Use ONE consistent scoring strategy
USE_COMBINED_SCORE = True  # Set to True for combined scoring, False for cascade
COMBINED_SCORING_STRATEGY = "combined"  # "combined", "cascade", or "sbert_only"
TFIDF_WEIGHT = 0.6
SBERT_WEIGHT = 0.4
COMBINED_THRESHOLD = 0.50

# SBERT settings
SBERT_MIN_SIM = 0.85  # Only used in cascade mode

# Negative penalty - FIXED: Much lower penalty
USE_NEGATIVE_PENALTY = True
NEGATIVE_PENALTY_ALPHA = 5.0  # 5% reduction per negative token (was 80%!)

# Zero-shot classifier settings
USE_ZERO_SHOT_PREFILTER = True
ZERO_SHOT_MODEL = "MoritzLaurer/deberta-v3-small-zeroshot-v1"  # Fast & accurate
ZERO_SHOT_THRESHOLD = 0.7

# ================ TEXT PROCESSING ================
USE_STEMMING = True
USE_STOPWORDS = True
NORMALIZE_HYPHENS = True
USE_CUSTOM_TFIDF_FEATURES = True

# ================ IMAGE SETTINGS ================
SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

# ================ DEBUGGING ================
ENABLE_DEBUG_PRINTS = True
SAVE_INTERMEDIATE_RESULTS = True
LOG_LEVEL = "INFO"
ENABLE_PDF_EXTRACTION = True

# ================ AUTOMATIC SETUP ================
def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PDF_CACHE_DIR, exist_ok=True)
    
    if SAVE_INTERMEDIATE_RESULTS:
        os.makedirs("logs", exist_ok=True)

# Call setup on import
setup_directories()
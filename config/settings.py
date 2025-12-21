"""
Centralized configuration for the quantum circuit image extractor.

This module defines file system locations, extraction limits, scoring
weights, and debug flags used throughout the pipeline. Importing the module
also ensures the expected cache/output/log directories exist via
``setup_directories``.
"""

import os

# ================ FILE PATHS ================
ID_FILE = "paper_list_36.txt"

# Image pipeline paths (kept identical to previous values for compatibility)
IMAGE_PIPELINE_OUTPUT_DIR = "images_extracted"
IMAGE_PIPELINE_CACHE_DIR = "arxiv_cache"  # Tar.gz source cache
IMAGE_PIPELINE_PDF_CACHE_DIR = "arxiv_pdf_cache"  # Extracted PDF cache (faster reaccess)

# Category cache path (moved out of arxiv_cache)
CATEGORY_CACHE_FILE = os.path.join("data", "arxiv_category_cache.json")

# Latex render pipeline paths
LATEX_LIVE_BLOCKS_ROOT = "circuit_images/live_blocks"
LATEX_BLOCKS_ROOT = "circuit_images/blocks"
LATEX_RENDER_DIR = "circuit_images/rendered_pdflatex"

# Backwards-compatible aliases (to avoid touching call sites yet)
OUTPUT_DIR = IMAGE_PIPELINE_OUTPUT_DIR
CACHE_DIR = IMAGE_PIPELINE_CACHE_DIR
PDF_CACHE_DIR = IMAGE_PIPELINE_PDF_CACHE_DIR
CATEGORY_CACHE = CATEGORY_CACHE_FILE

# ================ EXTRACTION LIMITS ================
MAX_IMAGES = 250
REQUEST_DELAY = 6  # seconds
TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5

# ================ SIMILARITY THRESHOLDS ================
SIMILARITY_THRESHOLD = 0.2  # TF-IDF gate (first filter) - raised for precision
SBERT_MIN_SIM = 0.85  # SBERT gate (optional, see USE_COMBINED_SCORE)

# Weighted combination scoring (recommended)
USE_COMBINED_SCORE = True  # If True, uses weighted combo; if False, uses cascade gates
APLHA = 0.6  # Adjusted alpha for combined scoring
TFIDF_WEIGHT = APLHA  # Weight for TF-IDF score (lexical precision)
SBERT_WEIGHT = 1- APLHA  # Weight for SBERT score (semantic understanding)
COMBINED_THRESHOLD = 0.55  # Combined score threshold for selection (raised to reduce false positives)

# Custom TF-IDF feature augmentation (can be toggled off)
USE_CUSTOM_TFIDF_FEATURES = True

USE_NEGATIVE_PENALTY = True
NEGATIVE_PENALTY_ALPHA = 70.0  # Percentage reduction per negative token (e.g., 5% per token, capped at 90%)

# ================ TEXT PROCESSING ================
USE_STEMMING = True
USE_STOPWORDS = True
NORMALIZE_HYPHENS = True

# YOLO - semantic information to mark the image then go to blip so that it knows where to look and extract text
# 

# ================ IMAGE SETTINGS ================
SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

# ================ DEBUGGING ================
ENABLE_DEBUG_PRINTS = True
SAVE_INTERMEDIATE_RESULTS = True
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
ENABLE_PDF_EXTRACTION = True  # Extract figures from PDF (Python-only, no external tools)

# ================ AUTOMATIC SETUP ================
def setup_directories():
    """Create necessary directories if they don't exist."""
    # Image pipeline directories
    os.makedirs(IMAGE_PIPELINE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_PIPELINE_CACHE_DIR, exist_ok=True)
    os.makedirs(IMAGE_PIPELINE_PDF_CACHE_DIR, exist_ok=True)

    # Category cache directory
    os.makedirs(os.path.dirname(CATEGORY_CACHE_FILE), exist_ok=True)

    # Latex render directories
    os.makedirs(LATEX_LIVE_BLOCKS_ROOT, exist_ok=True)
    os.makedirs(LATEX_BLOCKS_ROOT, exist_ok=True)
    os.makedirs(LATEX_RENDER_DIR, exist_ok=True)
    
    # Create logs directory if logging is enabled
    if SAVE_INTERMEDIATE_RESULTS:
        os.makedirs("logs", exist_ok=True)

# Call setup on import
setup_directories()
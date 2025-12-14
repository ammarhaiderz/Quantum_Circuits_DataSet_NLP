"""
Configuration settings for the SBERT-enhanced quantum circuit extractor.
"""
import os
from pathlib import Path

# =================================================
# PATH SETTINGS
# =================================================
BASE_DIR = Path(__file__).parent
ID_FILE = "paper_list_36.txt"
OUTPUT_DIR = "images_test_50_preproc_cached"
CACHE_DIR = "arxiv_cache"
MAX_IMAGES = 50

# =================================================
# REQUEST SETTINGS
# =================================================
REQUEST_DELAY = 15  # seconds

# =================================================
# SCORING SETTINGS
# =================================================
SIMILARITY_THRESHOLD = 0.5
# NOTE: USE_NEGATIVE_PENALTY and NEGATIVE_PENALTY_ALPHA are DEPRECATED.
# The new context-aware penalty system is now active (see HARD_REJECT_* and
# SOFT_PENALTY_WORDS below). These old variables are kept for reference only.
USE_NEGATIVE_PENALTY = True  # DEPRECATED: use context-aware penalty instead
NEGATIVE_PENALTY_ALPHA = 10.0  # DEPRECATED: use context-aware penalty instead
SBERT_MIN_SIM = 0.4
TOP_K_PER_PAPER = 10
PRINT_TOP_CAPTIONS = 5
# =================================================
# N-gram scoring configuration
NGRAM_WEIGHTS = {
    1: 1.0,    # Single word weight
    2: 1.5,    # Bigram weight
    3: 2.0,    # Trigram weight
    4: 2.5,    # 4-gram weight
}

# Penalty thresholds
PENALTY_THRESHOLD = 6.0  # Total penalty above which to reject
NEGATIVE_MULTIPLIER = 2.0  # Multiply negative scores

# Context window for n-grams
CONTEXT_WINDOW = 5  # Words to check around matches

# =================================================

# =================================================
# FILE SETTINGS
# =================================================
SUPPORTED_EXT = [".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"]

# =================================================
# QUERY SETS
# =================================================
QUERY_SETS = {
    "circuit_core": """
        quantum circuit diagram
        qubit circuit
        gate based quantum circuit
        gate sequence
        controlled gate
        qubit register
        circuit depth
    """,

    "gate_level": """
        cnot hadamard pauli
        rx ry rz
        controlled not
        control target qubit
        multi qubit gate
        ancilla qubit
    """,

    "algorithmic_circuits": """
        quantum algorithm 
        oracle
        grover 
        shor 
        qft
    """,

    "variational_circuits": """
        variational quantum circuit
        parameterized circuit
        ansatz circuit
        vqe circuit
        qaoa circuit
    """
}


# =================================================
# TEXT PREPROCESSING SETTINGS
# =================================================
USE_STEMMING = True
USE_STOPWORDS = True
NORMALIZE_HYPHENS = True

PROTECTED_TOKENS = {
    "cnot", "cx", "cz",
    "rx", "ry", "rz",
    "qft", "qaoa", "vqe", "vqc",
    "iswap"
}

QUANTUM_POSITIVE_TOKENS = {
    "quantum", "qubit", "qbit", "gate", "cnot", "hadamard", "pauli",
    "superposition", "entanglement", "coherence", "decoherence",
    "algorithm", "grover", "shor", "qft", "vqe", "qaoa", "ansatz",
    "rx", "ry", "rz", "swap", "iswap", "toffoli", "fredkin",
    "measurement", "observable", "expectation", "fidelity"
}

# =================================================
# NEGATIVE FILTERING SETTINGS
# =================================================
# ACTIVE PENALTY SYSTEM: Context-Aware Multi-Tier Filtering
# 
# Tier 1: Hard Rejection (check_hard_rejection)
#   - Exact phrase matching (HARD_REJECT_PHRASES)
#   - Category-based detection (HARD_REJECT_CATEGORIES)
#   - Result: score = 0.0 (absolute rejection)
#
# Tier 2: Soft Penalty (calculate_context_aware_penalty)
#   - Word-level weights (SOFT_PENALTY_WORDS)
#   - Context awareness: penalty reduced by 50% if quantum term nearby
#   - Result: score reduced by penalty amount
#
# See preprocessor.py for implementation details.

# Hard-rejection categories: if multiple keywords found + no quantum terms → reject
HARD_REJECT_CATEGORIES = {
    "electrical": {"voltage", "current", "resistor", "capacitor", "inductor", "transistor", "pcb", "board", "wire"},
    "digital": {"boolean", "logic", "gate", "processor", "cpu", "semiconductor"},
    "physical": {"3d", "mesh", "geometry", "solid", "rendering", "visualization"},
}

# Soft-penalty word weights with context sensitivity
# Negative values are penalties. Context-aware function reduces penalty if
# quantum term is found within 50 characters of the negative word.
SOFT_PENALTY_WORDS = {
    # Classical/non-quantum (high penalty)
    "classical": -3.0,
    "digital": -2.5,
    "electrical": -2.5,
    "analog": -2.5,

    # Electronics (medium penalty)
    "transistor": -2.0,
    "resistor": -2.0,
    "capacitor": -2.0,
    "inductor": -2.0,

    # Physical/visual (medium penalty)
    "3d": -2.0,
    "mesh": -1.5,
    "rendering": -1.5,

    # Lower-priority negatives (low penalty - context reduces further)
    "voltage": -1.0,
    "current": -1.0,
    "power": -1.0,
    "board": -1.0,
}

# Negative phrases with EXACT matching (high confidence negatives)
# If any of these exact phrases found → hard reject (score = 0.0)
HARD_REJECT_PHRASES = {
    "electrical circuit",
    "digital circuit",
    "classical circuit",
    "logic gate",
    "circuit board",
    "pcb design",
    "schematic symbol",
}

NEGATIVE_RAW_TOKENS = {
    # plots & charts
    "plot", "graph", "chart", "histogram",
    "scatter", "bar", "boxplot", "violin",
    "heatmap", "contour", "surface",
    "curve", "trend", "profil", "exampl",
    "implement", "flowchart", "demonstr",
    "code", "kernel", "notebook", "script", "function",
    "cuda", "cpu", "gpu", "illustration", "pulse", "duration", "scatter",
    "energy", "level", "spectrum", "eigenvalu", "matrix", "spectrum", "variational", "numerics",
    "fidelity", "overlap", "correlation", "concurrence", "log", "coefficient", "covariance",
    "dataset", "benchmark", "simulation", "simul", "iqm", "qpu", "hardware", "outlier",
    "training", "test", "validation", "fold", "cross-valid", "bloch", "sphere", "spherical", "spheric",
    "data", "dyson", "fit", "regress", "classif", "clust", "latice", "lattice", 
    "geometry", "graph", "network", "geometric", "time", "population", "ms","frequency", "domain", "duration", "mod", "modulus",
    "rate", "decay", "decoher", "nois", "signal", "volt", "current", "microsecond", "nanosecond", "millisecond",
    # statistics
    "distribut", "probabl", "expect",
    "varianc", "mean", "averag",
    "standard", "deviat", "confid",
    "interval", "percent", "ratio",
    "sparsity", "histogram", "bin", "sparse",
    # evaluation
    "result", "perform", "accuraci",
    "error", "loss", "benchmark",
    "metric", "score", "evaluat",
    "compar", "improv", "gain",
    # physics quantities
    "energi", "fidel", "overlap",
    "spectrum", "spectra",
    "eigenvalu", "eigenstat",
    "amplitud", "phase",
    "frequenc", "reson",
    # simulation
    "simul", "numer", "comput",
    "trial", "sampl",
    "iteration", "epoch",
    "converg", "optim",
    # experimental
    "measur", "readout",
    "nois", "decoher",
    "calibr", "volt",
    "current", "signal",
    # abstract visuals
    "node", "edge",
    "layout", "topolog",
    "network", "connect",
    # Electrical/electronic
    "resistor", "capacitor", "inductor", "transistor", "diode", "amplifier",
    "voltage", "current", "frequency", "signal", "impedance", "resistance",
    "capacitance", "inductance", "transmission", "power", "supply",
    "battery", "switch", "relay", "motor", "generator", "transformer",
    "analog", "digital", "pulse", "waveform", "amplitude", "phase",
    "ac", "dc", "alternating", "direct", "oscillator", "filter",
    "opamp", "operational", "mosfet", "bjt", "thyristor", "sensor",
    # 3D terms
    "3d", "three", "dimensional", "isometric", "perspective",
    "render", "rendering", "visualization", "volume", "mesh",
    "wireframe", "solid", "shaded", "lit", "lighting", "camera",
}

FILENAME_NEGATIVE_RAW = {
    "plot", "graph", "chart", "hist",
    "loss", "acc", "accuracy",
    "result", "results",
    "benchmark", "energy",
    "spectrum", "spectra",
    "prob", "distribution",
    "heatmap", "surface",
    "curve", "spectrum", "distribution",
    "simu", "simulation", "3d", "sphere", "spheric", "spherical",
    "duration", "time",
}

# =================================================
# SBERT SETTINGS
# =================================================
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
SBERT_BATCH_SIZE = 32
SBERT_NORMALIZE_EMBEDDINGS = True

# =================================================
# INITIALIZE DIRECTORIES
# =================================================
def initialize_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

initialize_directories()
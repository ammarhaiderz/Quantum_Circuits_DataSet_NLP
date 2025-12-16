"""
Configuration for circuit rendering pipeline.
"""

from pathlib import Path
from collections import defaultdict

# Paths
INPUT_TAR_FOLDER = Path("arxiv_cache")
OUTPUT_IMAGE_FOLDER = Path("rendered_circuits")
OUTPUT_METADATA_FILE = Path("circuit_metadata.json")
TEMP_FOLDER = Path("temp_latex")

# Circuit detection patterns (updated to capture labels)
CIRCUIT_PATTERNS = [
    # Complete \Qcircuit blocks with optional label before
    (r'(?:\\label\{[^}]+\}\s*)?\\Qcircuit\s*(?:@[A-Za-z]=[0-9\.]+(?:em|pt))?\s*\{.*?\}', 'qcircuit'),
    
    # Qcircuit environment with label
    (r'(?:\\label\{[^}]+\}\s*)?\\begin\{qcircuit\}.*?\\end\{qcircuit\}', 'qcircuit_env'),
    
    # Multi-line circuits
    (r'(?:\\label\{[^}]+\}\s*)?\\Qcircuit[^{]*\{[^}]*\\\\[^}]*\}', 'multiline_qcircuit'),
]

# LaTeX template for rendering (includes label if present)
LATEX_TEMPLATE = r"""
\documentclass[border=2pt]{{standalone}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{braket}}
\usepackage[%
  matrix,
  frame,
  arrow,
  xarrow,
  curve,
  arc,
  knot,
  braid,
  decorate,
  line,
 %
  circuitikz
]{{qcircuit}}

\begin{{document}}

{circuit_content}

\end{{document}}
"""

# Enhanced template that preserves labels
ENHANCED_LATEX_TEMPLATE = r"""
\documentclass[border=2pt]{{standalone}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{braket}}
\usepackage[%
  matrix,
  frame,
  arrow,
  xarrow,
  curve,
  arc,
  knot,
  braid,
  decorate,
  line,
 %
  circuitikz
]{{qcircuit}}

\begin{{document}}

{circuit_content}

\end{{document}}
"""

# Image settings
IMAGE_FORMAT = "png"
IMAGE_DPI = 300

# Label prefixes to categorize
LABEL_CATEGORIES = {
    'equation': ['eq:', 'equation:', 'eqn:', 'formula:'],
    'figure': ['fig:', 'figure:', 'scheme:'],
    'algorithm': ['algo:', 'algorithm:', 'protocol:', 'circuit:'],
    'section': ['sec:', 'section:', 'subsection:', 'subsec:'],
    'table': ['tab:', 'table:'],
}

# Parallel processing
MAX_WORKERS = 4
BATCH_SIZE = 10
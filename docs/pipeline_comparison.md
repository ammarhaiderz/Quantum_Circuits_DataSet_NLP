# Pipeline Comparison: Circuit Extraction vs Image Extraction

## Overview
The project has two parallel pipelines for extracting quantum-related content from arXiv papers:

1. **Circuit Extraction Pipeline** - Extracts and renders LaTeX circuit code blocks
2. **Image Extraction Pipeline** - Extracts general figures based on semantic similarity

Both now share similar metadata structures for consistency.

## Pipeline Comparison Table

| Aspect | Circuit Extraction | Image Extraction |
|--------|-------------------|------------------|
| **Purpose** | Extract LaTeX circuit code (tikz, circuitikz, qcircuit) | Extract general quantum figures (any format) |
| **Selection Method** | Pattern matching for circuit environments in LaTeX | TF-IDF + SBERT semantic similarity |
| **Source Material** | LaTeX source files only | LaTeX for captions + tarball images |
| **Output Format** | Rendered PNG/PDF from LaTeX code | Direct image extraction from tarball |
| **Output Location** | `circuit_images/live_blocks/rendered/` | `clean_images_50_preproc_cached/` |
| **Metadata Store** | `data/circuits.jsonl` + `circuits.json` | `data/images.jsonl` + `images.json` |
| **Code Entry Point** | `pipelines/latex_render/live_latex_extractor.py` | `pipelines/image_extraction/extraction_pipeline.py` |
| **Metadata Handler** | `core/circuit_store.py` | `core/image_extract_store.py` |

## Metadata Comparison

### Shared Fields (Both Pipelines)

| Field | Description | Notes |
|-------|-------------|-------|
| `arxiv_id` | Paper identifier | Always populated |
| `figure_number` | Figure number in paper | Parsed from caption |
| `page` | PDF page number | Best-effort (requires PDF text) |
| `quantum_problem` | Problem category | SBERT-based classification |
| `description` | List of text snippets | Caption + context from LaTeX |

### Circuit-Specific Fields

| Field | Description | Example |
|-------|-------------|---------|
| `gates` | List of quantum gates used | `["H", "CNOT", "Rz", "Measure"]` |
| `text_positions` | Character spans in PDF | `[[1234, 1456], [2345, 2567]]` |
| `latex_block` | Raw LaTeX code | `\begin{quantikz}...\end{quantikz}` |

### Image-Specific Fields (from ExtractedImage)

| Field | Description | Notes |
|-------|-------------|-------|
| `file_path` | Path to extracted image | Full absolute path |
| `img_name` | Image filename | e.g., `figure_1.png` |
| `similarity` | TF-IDF score | Raw + penalized scores |
| `sbert_sim` | SBERT similarity | Semantic similarity score |
| `combined_score` | Weighted combination | TF-IDF + SBERT |
| `best_query` | Best matching TF-IDF query | For debugging |
| `best_sbert_query` | Best matching SBERT query | For debugging |

## Selection Process Comparison

### Circuit Extraction

```
LaTeX Source
  ↓
Pattern Match (tikz, circuitikz, qcircuit)
  ↓
Extract Code Block
  ↓
Render via pdflatex
  ↓
Parse Gates (H, CNOT, etc.)
  ↓
Generate Metadata
```

**Key Characteristic**: High precision, low recall (only finds explicit circuit code)

### Image Extraction

```
LaTeX Source + Tarball
  ↓
Extract All Figures
  ↓
TF-IDF Filter (lexical similarity)
  ↓
SBERT Reranking (semantic similarity)
  ↓
Extract Images from Tarball
  ↓
Generate Metadata
```

**Key Characteristic**: Lower precision, higher recall (finds all relevant figures)

## Configuration Comparison

### Circuit Pipeline Config

```python
# From circuit_store.py / live_latex_extractor.py
LATEX_RENDER_DIR = "circuit_images/live_blocks"
USE_PDFLATEX = True
EXTRACT_GATES = True
```

### Image Pipeline Config

```python
# From config/settings.py
MAX_IMAGES = 250
TOP_K_PER_PAPER = 5
SIMILARITY_THRESHOLD = 0.08  # TF-IDF gate
SBERT_MIN_SIM = 0.3         # SBERT gate
USE_COMBINED_SCORE = True
TFIDF_WEIGHT = 0.3
SBERT_WEIGHT = 0.7
```

## Use Case Recommendations

### Use Circuit Extraction When:
- Need actual quantum gate sequences
- Want to analyze circuit structure programmatically
- Need rendered circuit diagrams
- Working with LaTeX-based quantum computing papers

### Use Image Extraction When:
- Need diverse figure types (not just circuits)
- Want semantic similarity matching
- Need to capture figures that aren't LaTeX-based
- Working with mixed-format papers (images embedded in LaTeX)

## Shared Infrastructure

Both pipelines share:
- PDF handling (PyMuPDF)
- LaTeX text utilities
- Caption normalization
- Page number detection
- Quantum problem classification (SBERT)
- JSONL + JSON dual output format

## Output Examples

### Circuit Extraction Output (`circuits.json`)

```json
{
  "2301.01234": {
    "1": {
      "arxiv_id": "2301.01234",
      "figure_number": 1,
      "page": 3,
      "gates": ["H", "CNOT", "Measure"],
      "quantum_problem": "Quantum Algorithms",
      "descriptions": [
        "Bell state preparation circuit.",
        "The circuit creates maximally entangled states..."
      ],
      "text_positions": [[1234, 1456], [2345, 2567]]
    }
  }
}
```

### Image Extraction Output (`images.json`)

```json
{
  "2301.01234": {
    "1": {
      "arxiv_id": "2301.01234",
      "figure_number": 1,
      "page": 3,
      "quantum_problem": "Quantum Algorithms",
      "description": [
        "Fig. 1: Bell state preparation.",
        "The circuit creates maximally entangled states..."
      ]
    }
  }
}
```

Note: Image extraction metadata doesn't include `gates` or `text_positions` (those are circuit-specific).

## Code Reuse Summary

Functions repurposed from `circuit_store.py` → `image_extract_store.py`:

- `find_caption_page_in_pdf()` - Page number detection
- `classify_quantum_problem()` - SBERT-based categorization
- `normalize_caption_text()` - LaTeX cleanup
- `_extract_paragraph_after_figure()` - Context extraction

Functions specific to circuit extraction (not repurposed):

- `_parse_gates_from_latex()` - Gate sequence extraction
- `_align_description_items_to_pdf_spans()` - Character offset alignment
- `locate_figure_block()` - Figure region detection

## Performance Characteristics

| Metric | Circuit Extraction | Image Extraction |
|--------|-------------------|------------------|
| **Speed** | Slower (requires LaTeX compilation) | Faster (direct extraction) |
| **Precision** | Very high (~95%) | Medium-high (~70-80%) |
| **Recall** | Low-medium (only LaTeX circuits) | High (all relevant figures) |
| **Output Quality** | Consistent (rendered from code) | Variable (depends on source) |

## When to Run Both Pipelines

Run both when you need:
- Comprehensive quantum figure collection
- Circuit-specific metadata (gates, etc.)
- Comparison between LaTeX-based and general figures
- Maximum coverage across different paper formats

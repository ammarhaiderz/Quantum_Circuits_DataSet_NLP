# Quantum Dataset NLP Pipelines

This project extracts quantum-circuit figures and LaTeX circuit blocks from arXiv sources, scores them (TF-IDF + SBERT), and renders LaTeX circuits when needed.

## What it does
- Download arXiv sources/PDFs, extract figures and captions, filter with TF-IDF + SBERT, and save the best images.
- Detect LaTeX circuit blocks (Qcircuit/quantikz), render them to PDF/PNG, and log metadata.
- Keep circuit metadata in JSON/JSONL for downstream use.

## Pipelines
### Image Extraction
- Reads arXiv IDs, downloads source/PDF, parses figures and captions.
- Filters candidates with TF-IDF, reranks with SBERT, applies combined score/thresholds.
- Saves top images to `IMAGE_PIPELINE_OUTPUT_DIR` and records metadata for analysis.

### LaTeX Render (Qcircuit/quantikz)
- Scans LaTeX for circuit blocks (Qcircuit/quantikz), including live extraction during image processing.
- Saves raw blocks under `LATEX_LIVE_BLOCKS_ROOT`/`LATEX_BLOCKS_ROOT`, renders to PDF/PNG in `LATEX_RENDER_DIR`.
- Emits circuit metadata (gates, descriptions, file refs) into `core/circuit_store` JSON/JSONL for downstream use.

## Project layout
- Entry: `main.py` → `pipelines.image_extraction.runner.run`
- Image pipeline: `pipelines/image_extraction/{runner,extraction_pipeline,image_extractor,tfidf_filter,sbert_reranker}`
- LaTeX render helpers: `pipelines/latex_render/{live_latex_extractor,latex_utils_emb}`
- Shared utilities: `shared/{figure_data,logging_utils,file_utils,preprocessor,arxiv_validator}`
- Core services: `core/{circuit_store,quantum_problem_classifier}`
- Config: `config/{settings,queries,quantum_problem_labels}`

## Setup
Use the bundled virtualenv:
```
D:/repositories/Github-NLP/Quantum_Dataset_NLP/nlp/Scripts/python.exe -m pip install --upgrade pip
D:/repositories/Github-NLP/Quantum_Dataset_NLP/nlp/Scripts/python.exe -m pip install -r requirements.txt
```

## Configuration
Edit `config/settings.py` for paths and thresholds:
- Image pipeline paths: `IMAGE_PIPELINE_OUTPUT_DIR`, `IMAGE_PIPELINE_CACHE_DIR`, `IMAGE_PIPELINE_PDF_CACHE_DIR`
- LaTeX render paths: `LATEX_LIVE_BLOCKS_ROOT`, `LATEX_BLOCKS_ROOT`, `LATEX_RENDER_DIR`
- Scoring/limits: `MAX_IMAGES`, `TOP_K_PER_PAPER`, `SIMILARITY_THRESHOLD`, `SBERT_MIN_SIM`, `USE_COMBINED_SCORE`, `TFIDF_WEIGHT`, `SBERT_WEIGHT`, `COMBINED_THRESHOLD`
- Text processing: `USE_STEMMING`, `USE_STOPWORDS`, `NORMALIZE_HYPHENS`, `USE_CUSTOM_TFIDF_FEATURES`, `USE_NEGATIVE_PENALTY`

The default ID list is `paper_list_36.txt` (one arXiv ID per line). Update it as needed.

## Running
```
D:/repositories/Github-NLP/Quantum_Dataset_NLP/nlp/Scripts/python.exe ./main.py
```
Outputs land in `IMAGE_PIPELINE_OUTPUT_DIR`; LaTeX renders go to `LATEX_RENDER_DIR` and `LATEX_LIVE_BLOCKS_ROOT`/`LATEX_BLOCKS_ROOT`.

## Tests
Run the full suite:
```
D:/repositories/Github-NLP/Quantum_Dataset_NLP/nlp/Scripts/python.exe -m pytest
```

## Notes
- All pipeline code now lives under `pipelines/`; the legacy `pipeline/` folder was removed.
- Core uses the per-pipeline paths from `config/settings.py`.
- Rendering LaTeX blocks requires `pdflatex` available on PATH; PyMuPDF (`fitz`) is used for PDF→PNG when present.

# Pipeline Separation Plan (no code changes yet)

Goal: separate image-extraction and latex-render pipelines with minimal risk. Code stays exact until each planned step is executed and verified.

## Step 0: Safety/Scope
- Create/confirm working branch: `git checkout -b pipeline-reorg` (already on Ammar_Pipeline if preferred).
- No code moves yet; plan only.

## Step 1: Add shared package (structure only)
- Create `shared/` with `__init__.py` (empty) to prepare for moves.
- No file moves yet. 
- Verification: `python -m compileall shared` (should be trivial) and run `python -m pip show` not needed; ensure main pipeline still runs: `python main.py --help` (or basic run) — expect no behavior change.

## Step 2: Move truly shared modules
- Move (without edits) to `shared/`: `models/figure_data.py`, `utils/logging_utils.py`, `utils/file_utils.py`, `core/preprocessor.py`, `core/arxiv_validator.py` (if used by both).
- Update imports in both pipelines to `from shared...` (mechanical rename only; no logic edits).
- Verification: run unit/integration tests touching both pipelines: `pytest tests/test_integration.py tests/test_latex_render.py` (or minimal smoke: `python main.py` with a small ID list; run any live-latex smoke if available).

## Step 3: Isolate image-extraction pipeline
- Create `pipelines/image_extraction/` package; move `pipeline/extraction_pipeline.py`, `core/image_extractor.py`, `core/tfidf_filter.py`, `core/sbert_reranker.py`, and the main runner (`main.py` contents) into `pipelines/image_extraction/runner.py`.
- Adjust imports to new locations; keep logic identical.
- Leave a thin top-level `main.py` that delegates to the runner (no behavioral change).
- Verification: run `python main.py` end-to-end with a small ID list; ensure caches/outputs still land in expected dirs.

## Step 4: Isolate latex-render pipeline
- Create `pipelines/latex_render/` package; move `core/live_latex_extractor.py` and `utils/latex_utils_emb.py` here (plus any render-only helpers).
- Adjust imports in render tooling/tests to new paths; keep logic identical.
- Verification: run render smoke tests: `pytest tests/test_latex_render.py` (or the live render command if any). Ensure rendered PDFs/PNGs still appear in the configured folders.

## Step 5: Separate outputs/config knobs
- In `config/settings.py`, add per-pipeline output path variables (e.g., `IMAGE_PIPELINE_OUTPUT_DIR`, `LATEX_RENDER_OUTPUT_DIR`) and wire callers to use them (path rewiring only, no logic change).
- Ensure `setup_directories()` creates both sets. 
- Verification: re-run both pipelines; confirm artifacts land in new scoped folders and no regressions.

## Step 6: Tests & cleanup
- Run full test suite: `pytest`.
- Update documentation/README to reflect new entry points and folder layout.
- Remove any obsolete paths/import aliases.

Notes:
- After each step, commit the minimal change and run the listed verification before proceeding.
- Keep logic identical; only paths/imports change until Step 5 (path variables).
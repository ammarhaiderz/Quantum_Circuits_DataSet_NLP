# Pipeline Separation Plan (working log)

Goal: keep image-extraction and latex-render pipelines cleanly separated with minimal behavior change.

## Completed steps
- Step 1: Added `shared/` package scaffold.
- Step 2: Moved common modules to `shared/` and rewired imports (figure_data, logging_utils, file_utils, preprocessor, arxiv_validator).
- Step 3: Isolated image-extraction pipeline under `pipelines/image_extraction/`; `main.py` now delegates to its runner; legacy `pipeline/` folder removed.
- Step 4: Isolated latex-render tools under `pipelines/latex_render/` (moved `live_latex_extractor.py` and `latex_utils_emb.py`); image pipeline now imports render helpers from the new package; `tests/test_latex_render.py` passes.

## Current layout (high level)
- Entry: `main.py` → `pipelines.image_extraction.runner.run`
- Image pipeline: `pipelines/image_extraction/{runner,extraction_pipeline,image_extractor,tfidf_filter,sbert_reranker}`
- Latex render pipeline: `pipelines/latex_render/{live_latex_extractor,latex_utils_emb}`
- Shared: `shared/{figure_data,logging_utils,file_utils,preprocessor,arxiv_validator}`
- Core (left): `core/{circuit_store,quantum_problem_classifier}`

## Next steps (planned)
1) Step 5: Split output/config knobs
	- Add per-pipeline paths in `config/settings.py` (e.g., `IMAGE_PIPELINE_OUTPUT_DIR`, `LATEX_RENDER_OUTPUT_DIR`, cache/log dirs if needed).
	- Wire image pipeline and latex-render helpers to use the new constants; keep behavior identical except path targets.
	- Ensure `setup_directories()` prepares both sets.
	- Verify with `python main.py` (smoke) and `pytest tests/test_latex_render.py`.

2) Step 6: Final cleanup/tests
	- Run full `pytest`.
	- Update docs/README to note new entry points and package layout.
	- Remove any stale references (if any appear) and commit.

Keep this file as the anchor for progress/status.
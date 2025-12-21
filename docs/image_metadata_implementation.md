# Image Metadata Enrichment - Implementation Summary

## Overview
Created a new metadata enrichment system for the image extraction pipeline by repurposing functions from `circuit_store.py`. The system generates standardized metadata for all extracted images, mirroring the structure used in the circuit extraction pipeline.

## Files Created/Modified

### New Files
1. **`core/image_extract_store.py`** (237 lines)
   - Main metadata generation module for image extraction pipeline
   - Repurposes functions from `circuit_store.py` for general figure processing
   - Handles JSONL append-only logging and JSON map generation

2. **`test_image_metadata.py`** (141 lines)
   - Comprehensive test suite for metadata generation
   - Tests figure number parsing, metadata generation, and output finalization
   - All tests passing ✓

### Modified Files
1. **`pipeline/extraction_pipeline.py`**
   - Added import: `from core.image_extract_store import generate_image_metadata, emit_image_record`
   - Added metadata generation after image extraction (lines 405-414)
   - Generates metadata for each extracted image and writes to JSONL

2. **`pipelines/image_extraction/runner.py`**
   - Added import: `from core.image_extract_store import finalize_images_output`
   - Added finalization call to regenerate `images.json` at pipeline completion

## Metadata Fields Generated

For each extracted image, the system generates:

| Field | Type | Description | Status |
|-------|------|-------------|--------|
| `arxiv_id` | str | Paper identifier | ✓ Always populated |
| `figure_number` | int | Figure number in paper | ✓ Extracted from caption |
| `page` | int | PDF page number | ⚠️ Best-effort (requires PDF text layer) |
| `quantum_problem` | str | Problem category | ⚠️ Requires SBERT model |
| `description` | list[str] | Caption + context snippets | ✓ Always populated |

### Status Legend
- ✓ Always populated with valid data
- ⚠️ Populated when possible, null otherwise

## Key Functions

### From `core/image_extract_store.py`

```python
def generate_image_metadata(arxiv_id: str, caption: str, preprocessed_text: str) -> dict
```
Main entry point - generates complete metadata dictionary.

```python
def emit_image_record(metadata: dict) -> None
```
Appends metadata to `data/images.jsonl` (append-only log).

```python
def finalize_images_output() -> None
```
Regenerates `data/images.json` from JSONL (structured mapping by arxiv_id and figure_number).

### Repurposed Functions (imported from `circuit_store.py`)

- `find_caption_page_in_pdf()` - Locates page number via PDF text search
- `classify_quantum_problem()` - SBERT-based problem categorization
- `normalize_caption_text()` - Cleans LaTeX artifacts from captions
- `_extract_paragraph_after_figure()` - Extracts context paragraph from LaTeX

## Output Format

### JSONL Format (`data/images.jsonl`)
Append-only log, one JSON object per line:
```json
{"arxiv_id": "2301.01234", "figure_number": 1, "page": 3, "quantum_problem": "Error Correction", "description": ["Fig. 1: Quantum error correction circuit.", "The circuit implements..."]}
```

### JSON Format (`data/images.json`)
Nested mapping for easy lookup:
```json
{
  "2301.01234": {
    "1": {
      "arxiv_id": "2301.01234",
      "figure_number": 1,
      "page": 3,
      "quantum_problem": "Error Correction",
      "description": ["Fig. 1: Quantum error correction circuit.", "The circuit implements..."]
    }
  }
}
```

## Integration Flow

```
ExtractionPipeline.process_paper()
  ↓
Extract images (ImageExtractor.extract_images())
  ↓
For each ExtractedImage:
  ↓
  generate_image_metadata() ← Repurposes circuit_store.py functions
    - Parse figure number from caption
    - Find page in PDF (via PyMuPDF)
    - Classify quantum problem (via SBERT)
    - Build description list (caption + context)
  ↓
  emit_image_record() ← Append to images.jsonl
  ↓
Continue to next paper...
  ↓
After all papers:
  ↓
finalize_images_output() ← Regenerate images.json from JSONL
```

## Testing Results

All tests passing ✓

### Test Coverage
- ✓ Figure number parsing from various caption formats
- ✓ Complete metadata generation with null handling
- ✓ Description list building
- ✓ JSONL emission and JSON finalization
- ✓ Multi-paper, multi-figure scenarios

### Sample Test Output
```
=== Testing Figure Number Parsing ===
✓ Caption: 'Fig. 5: Quantum circuit...' -> 5 (expected 5)
✓ Caption: 'Figure 12: NCV quantum gate...' -> 12 (expected 12)
✓ Caption: 'Circuit diagram...' -> None (expected None)

=== Testing Emit and Finalize ===
✓ Generated images.json with 2 papers:
  2301.01234: 2 figures
  2302.05678: 1 figures
```

## Usage

The system runs automatically during pipeline execution. No manual intervention needed.

### Manual Metadata Generation (if needed)
```python
from core.image_extract_store import generate_image_metadata, emit_image_record

metadata = generate_image_metadata(
    arxiv_id="2301.01234",
    caption="Fig. 5: Quantum circuit for error correction.",
    preprocessed_text="quantum circuit error correction"
)

emit_image_record(metadata)
```

### Regenerate JSON from JSONL
```python
from core.image_extract_store import finalize_images_output

finalize_images_output()  # Regenerates data/images.json
```

## Limitations

1. **Page numbers**: Dependent on PDF text layer quality. Short label-like captions (e.g., "NCV gate library") often rendered as graphics and won't be found in text layer.

2. **Quantum problem classification**: Requires SBERT model to be loaded. Falls back to `null` if unavailable.

3. **Description context**: Best results when LaTeX source is available. Falls back to caption only when source download fails.

## Future Enhancements (Optional)

- Add `text_positions` (character spans in PDF) - requires per-sentence alignment
- Add `figure_block_span` - approximate figure region in PDF
- OCR fallback for figure-embedded text that's missing from PDF text layer
- More robust context extraction from LaTeX (multi-paragraph support)

## Files Generated

- `data/images.jsonl` - Append-only log of all image metadata
- `data/images.json` - Structured mapping for easy lookup
- Both files created automatically by pipeline

## Compatibility

✓ Works with existing image extraction pipeline
✓ No breaking changes to existing functionality
✓ Metadata generation is additive (doesn't modify `ExtractedImage` dataclass)
✓ Graceful degradation when optional features unavailable (page, quantum_problem)

"""Metadata enrichment utilities for image extraction pipeline.

This module provides functions to generate metadata for extracted images,
repurposing functionality from circuit_store.py but optimized for the
image extraction pipeline which processes general figures (not just circuits).

Generated metadata includes:
- arxiv_id: Paper identifier
- figure_number: Figure number in the paper
- page: PDF page where the figure appears
- quantum_problem: Classified problem category (e.g., "Error Correction", "Quantum Algorithms")
- description: List of descriptive text snippets (caption + context)
"""

import json
from pathlib import Path
import re
from typing import Optional

try:
    import fitz
except Exception:
    fitz = None

from config.settings import (
    IMAGE_PIPELINE_CACHE_DIR,
    IMAGE_PIPELINE_PDF_CACHE_DIR,
    LATEX_RENDER_DIR,
)

from core.circuit_store import (
    find_caption_page_in_pdf,
    classify_quantum_problem,
    normalize_caption_text,
    _extract_paragraph_after_figure,
)

from utils.latex_text_utils import (
    load_latex_source,
    extract_context_snippet,
)


DATA_DIR = Path('data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_JSONL_PATH = DATA_DIR / 'images.jsonl'
IMAGES_JSON_PATH = DATA_DIR / 'images.json'


def _ensure_data_files() -> None:
    """Create empty data files if missing."""
    if not IMAGES_JSONL_PATH.exists():
        IMAGES_JSONL_PATH.write_text('', encoding='utf-8')
    if not IMAGES_JSON_PATH.exists():
        IMAGES_JSON_PATH.write_text('{}', encoding='utf-8')


_ensure_data_files()


def _parse_figure_number_from_caption(caption: str) -> Optional[int]:
    """Extract figure number from caption text.
    
    Parameters
    ----------
    caption : str
        Caption text that may contain figure references like "Fig. 3" or "Figure 12".
    
    Returns
    -------
    int or None
        Extracted figure number, or None if not found.
    """
    if not caption:
        return None
    
    # Look for patterns like "Fig. 3", "Figure 12", etc.
    patterns = [
        r'\bfig\.?\s*(\d+)\b',
        r'\bfigure\s+(\d+)\b',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, caption, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


def _build_description_list(
    caption: str,
    latex_source: Optional[str] = None,
) -> list[str]:
    """Build description list from caption and LaTeX context.
    
    Parameters
    ----------
    caption : str
        Figure caption text.
    latex_source : str, optional
        Full LaTeX source of the paper.
    
    Returns
    -------
    list[str]
        List of description strings, starting with the caption.
    """
    descriptions = []
    
    # 1. Add normalized caption as first item
    normalized_caption = normalize_caption_text(caption)
    if normalized_caption:
        descriptions.append(normalized_caption)
    
    # 2. Try to extract paragraph after figure from LaTeX source
    if latex_source and caption:
        try:
            para = _extract_paragraph_after_figure(latex_source, caption)
            if para and para != normalized_caption:
                descriptions.append(para)
        except Exception:
            pass
    
    # 3. If we still have only caption, try to extract context snippet
    if len(descriptions) == 1 and latex_source and caption:
        try:
            snippet = extract_context_snippet(latex_source, caption, max_length=300)
            if snippet and snippet != normalized_caption:
                descriptions.append(snippet)
        except Exception:
            pass
    
    return descriptions


def generate_image_metadata(
    arxiv_id: str,
    caption: str,
    preprocessed_text: str,
    img_name: Optional[str] = None,
) -> dict:
    """Generate metadata for an extracted image.
    
    Parameters
    ----------
    arxiv_id : str
        arXiv paper identifier.
    caption : str
        Figure caption text.
    preprocessed_text : str
        Preprocessed caption text (used for quantum problem classification).
    img_name : str, optional
        Image filename (used as unique key in JSON mapping).
    
    Returns
    -------
    dict
        Dictionary containing:
        - arxiv_id: Paper identifier
        - figure_number: Parsed from caption (or None)
        - img_name: Image filename
        - page: PDF page number (or None)
        - quantum_problem: Classified problem category (or None)
        - description: List of descriptive text snippets
    """
    metadata = {
        "arxiv_id": arxiv_id,
        "figure_number": None,
        "img_name": img_name,
        "page": None,
        "quantum_problem": None,
        "description": [],
    }
    
    # 1. Parse figure number from caption
    figure_number = _parse_figure_number_from_caption(caption)
    metadata["figure_number"] = figure_number
    
    # 2. Find page number in PDF
    try:
        result = find_caption_page_in_pdf(arxiv_id, caption)
        if result:
            page_num, detected_fig_num = result
            metadata["page"] = page_num
            # Update figure_number if detected from PDF
            if detected_fig_num is not None and figure_number is None:
                metadata["figure_number"] = detected_fig_num
    except Exception:
        pass
    
    # 3. Classify quantum problem using preprocessed text
    try:
        problem_type = classify_quantum_problem(preprocessed_text)
        metadata["quantum_problem"] = problem_type
    except Exception:
        pass
    
    # 4. Build description list (caption + context)
    try:
        # Try to load LaTeX source for context extraction
        latex_source = None
        try:
            latex_source = load_latex_source(arxiv_id)
        except Exception:
            pass
        
        descriptions = _build_description_list(caption, latex_source)
        metadata["description"] = descriptions
    except Exception:
        # Fallback: just use normalized caption
        try:
            normalized = normalize_caption_text(caption)
            if normalized:
                metadata["description"] = [normalized]
        except Exception:
            metadata["description"] = [caption]
    
    return metadata


def emit_image_record(metadata: dict) -> None:
    """Append an image metadata record to images.jsonl.
    
    Parameters
    ----------
    metadata : dict
        Image metadata dictionary from generate_image_metadata().
    
    Returns
    -------
    None
    """
    try:
        with open(IMAGES_JSONL_PATH, 'a', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"Warning: Failed to write image record: {e}")


def _regenerate_images_json() -> None:
    """Regenerate images.json from images.jsonl.
    
    Creates a mapping: {img_name: metadata} (same structure as circuits.json)
    """
    if not IMAGES_JSONL_PATH.exists():
        return
    
    records_dict = {}
    
    try:
        with open(IMAGES_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    img_name = record.get("img_name")
                    
                    if not img_name:
                        continue
                    
                    # Use img_name as the top-level key (same as circuits.json)
                    records_dict[img_name] = record
                    
                except Exception:
                    continue
        
        with open(IMAGES_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(records_dict, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Warning: Failed to regenerate images.json: {e}")


def finalize_images_output() -> None:
    """Regenerate images.json from the JSONL log.
    
    Call this at the end of pipeline processing to create the consolidated output.
    """
    _regenerate_images_json()


if __name__ == "__main__":
    # Test metadata generation
    test_arxiv_id = "2301.01234"
    test_caption = "Fig. 5: Quantum circuit for error correction using Shor's 9-qubit code."
    test_preprocessed = "quantum circuit error correction shor 9 qubit code"
    
    metadata = generate_image_metadata(test_arxiv_id, test_caption, test_preprocessed)
    print("Generated metadata:")
    print(json.dumps(metadata, indent=2))

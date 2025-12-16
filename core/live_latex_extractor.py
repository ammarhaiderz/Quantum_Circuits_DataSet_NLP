#!/usr/bin/env python3
"""Extract live LaTeX Qcircuit blocks from text files and optionally render them.
"""
from pathlib import Path
import json
import argparse
import sys
from core.latex_extractor import render_saved_blocks_with_pdflatex_module
import shutil
from typing import Optional
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
import re

def extract_qcircuit_blocks_from_text(text: str):
    """Fallback extractor copied from the project's template.

    Returns a list of unique blocks as strings.
    """
    blocks = []
    start_pattern = re.compile(r'(?:\\label\{[^}]*\}\s*)?\\Qcircuit\b', re.IGNORECASE)

    for m in start_pattern.finditer(text):
        start_idx = m.start()
        brace_idx = text.find('{', m.end())
        if brace_idx == -1:
            continue

        depth = 0
        i = brace_idx
        while i < len(text):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    block = text[start_idx:i+1]
                    if re.search(r'\\(qw|gate|ctrl|targ|meter|lstick|rstick|cw)\b', block):
                        blocks.append(block)
                    break
            i += 1

    seen = set()
    unique = []
    for b in blocks:
        key = b.strip()[:300]
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique



def _safe_name(name: str) -> str:
    return name.replace('/', '__').replace('..', '__')


def process_text(text: str, source_name: str = 'inline', out_root: str = 'circuit_images/live_blocks', render: bool = True, render_with_module: bool = False):
    """Extract blocks from `text` and save them under `out_root/source_name`.

    Returns a dict with summary information.
    """
    out_root = Path(out_root)
    dest = out_root / _safe_name(source_name)
    raw_dir = dest / 'raw_blocks'
    dest.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    blocks = extract_qcircuit_blocks_from_text(text)
    record = {
        'source': source_name,
        'blocks_count': len(blocks),
        'blocks': [],
    }

    for i, block in enumerate(blocks):
        name = f"{_safe_name(source_name)}_block_{i:03d}.tex"
        (raw_dir / name).write_text(block, encoding='utf-8')
        record['blocks'].append({'index': i, 'block_file': str((raw_dir / name).relative_to(out_root)), 'length': len(block)})

    summary_path = dest / 'summary.json'
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')

    if render:
        try:
            # Render only the current paper's blocks (dest contains this paper's raw_blocks)
            rendered_dir = dest / 'rendered'
            render_saved_blocks_with_pdflatex_module(blocks_root=str(dest), out_dir=str(rendered_dir))

            # Also copy rendered PDFs into a common folder with a paper-specific prefix
            common_dir = Path('circuit_images/rendered_pdflatex')
            common_dir.mkdir(parents=True, exist_ok=True)

            safe_src = _safe_name(source_name)
            if rendered_dir.exists():
                for f in rendered_dir.iterdir():
                    if f.is_file() and f.suffix.lower() == '.pdf':
                        try:
                            dest_name = f"{safe_src}__{f.name}"
                            shutil.copyfile(f, common_dir / dest_name)
                            # Convert per-paper PDF to PNG (same folder)
                            try:
                                png_out = f.with_suffix('.png')
                                _pdf_to_png(f, png_out)
                            except Exception:
                                pass
                            # Convert copied common PDF to PNG
                            try:
                                common_pdf = common_dir / dest_name
                                common_png = common_pdf.with_suffix('.png')
                                _pdf_to_png(common_pdf, common_png)
                            except Exception:
                                pass
                        except Exception:
                            # Non-fatal; continue copying other files
                            pass

        except ImportError:
            pass

    return record


def process_file(path: str, out_root: str = 'circuit_images/live_blocks', render: bool = True, render_with_module: bool = False):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    raw = p.read_bytes()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('latin-1', errors='ignore')

    return process_text(text, source_name=p.name, out_root=out_root, render=render, render_with_module=render_with_module)


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> bool:
    """Convert a single-page PDF to PNG.
    Uses PyMuPDF (fitz) if available. Returns True on success.
    """
    if fitz is None:
        return False

    try:
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(0)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(png_path))
        doc.close()
        return True
    except Exception:
        try:
            doc.close()
        except Exception:
            pass
        return False


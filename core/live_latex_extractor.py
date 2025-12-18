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
import hashlib
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
import re
import time

# Global set to track processed block content hashes (prevents duplicates)
_PROCESSED_BLOCKS = set()

def extract_qcircuit_blocks_from_text(text: str):
    """Extract both Qcircuit and quantikz circuit blocks from text.

    Returns a list of unique blocks as strings.
    """
    blocks = []
    def _line_is_commented(txt: str, idx: int) -> bool:
        """Return True if the position `idx` lies on a line that is commented out
        (i.e. there is an unescaped `%` between the start of the line and idx).
        """
        # find start of the line
        ls = txt.rfind('\n', 0, idx)
        start = ls + 1 if ls != -1 else 0
        segment = txt[start:idx]
        i = 0
        while True:
            p = segment.find('%', i)
            if p == -1:
                return False
            # count backslashes before % to see if escaped
            back = 0
            j = p - 1
            while j >= 0 and segment[j] == '\\':
                back += 1
                j -= 1
            if back % 2 == 0:
                return True
            i = p + 1
    
    # Extract Qcircuit blocks
    qcircuit_pattern = re.compile(r'(?:\\label\{[^}]*\}\s*)?\\Qcircuit\b', re.IGNORECASE)
    for m in qcircuit_pattern.finditer(text):
        start_idx = m.start()
        if _line_is_commented(text, start_idx):
            continue
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
    
    # Extract quantikz blocks (environment-based)
    quantikz_pattern = re.compile(r'\\begin\{quantikz\}.*?\\end\{quantikz\}', re.DOTALL | re.IGNORECASE)
    for m in quantikz_pattern.finditer(text):
        start_idx = m.start()
        if _line_is_commented(text, start_idx):
            continue
        block = m.group(0)
        # Verify it has circuit elements
        if re.search(r'\\(qw|gate|ctrl|targ|meter|lstick|rstick|gategroup)\b', block):
            blocks.append(block)

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


def _extract_gates_from_block(block: str):
    r"""Extract a list of gate names from a LaTeX circuit block.

    This uses several heuristics:
    - Find `\gate{...}` contents and normalize tokens.
    - Detect `\ctrl`, `\targ`, `\swap`, `\meter` etc.
    - Fallback: find textual gate names (H, X, Y, Z, CNOT, CX, Toffoli, SWAP, Rz, Rx).
    Returns a unique list of uppercase gate tokens.
    """
    gates = []
    try:
        # 1) \\gate{...}
        for m in re.findall(r"\\\\gate\{([^}]*)\}", block):
            # split on non-alphanumeric to get tokens like H, X, R_z(\theta)
            parts = re.split(r"[^A-Za-z0-9_\\]+", m)
            for p in parts:
                if not p:
                    continue
                # normalize common forms
                token = p.strip()
                token_upper = token.upper()
                # Map common names
                if token_upper in ("HADAMARD",):
                    gates.append('H')
                elif token_upper in ("CNOT", "CX"):
                    gates.append('CNOT')
                elif token_upper in ("TOFFOLI", "CCX"):
                    gates.append('TOFFOLI')
                elif token_upper.startswith('R') and any(ch.isdigit() or ch in 'ZXY' for ch in token_upper):
                    # RZ, RX, RY, R1, R2 -> keep prefix
                    gates.append(token_upper)
                else:
                    # Single-letter common gates
                    if token_upper in ('H', 'X', 'Y', 'Z', 'S', 'T'):
                        gates.append(token_upper)
                    else:
                        # generic token
                        gates.append(token_upper)

        # 2) detect ctrl/targ pattern -> CNOT
        if re.search(r"\\\\ctrl\b", block) and re.search(r"\\\\targ\b", block):
            gates.append('CNOT')

        # 3) detect swap
        if re.search(r"\\\\swap\b", block) or re.search(r"\\\\qswap\b", block):
            gates.append('SWAP')

        # 4) detect meter/measure
        if re.search(r"\\\\meter\b|\\\\measure\b", block):
            gates.append('MEASURE')

        # 5) detect targs alone
        if re.search(r"\\\\targ\b", block) and 'CNOT' not in gates:
            gates.append('TARG')

        # 6) textual fallbacks: look for known gate names
        for txt in re.findall(r"\b(H|X|Y|Z|CNOT|CX|SWAP|TOFFOLI|CCX|S|T|RZ|RX|RY)\b", block, flags=re.IGNORECASE):
            gates.append(txt.upper())

    except Exception:
        pass

    # normalize unique preserving order
    seen = set()
    out = []
    for g in gates:
        if not g:
            continue
        g2 = g.upper()
        if g2 not in seen:
            seen.add(g2)
            out.append(g2)
    return out


def process_text(text: str, source_name: str = 'inline', out_root: str = 'circuit_images/live_blocks', render: bool = True, render_with_module: bool = False, arxiv_id: Optional[str] = None, start_figure_num: Optional[int] = None, caption_text: Optional[str] = None, figure_number: Optional[int] = None, panel: Optional[int] = None, figure_label: Optional[str] = None):
    """Extract blocks from `text` and save them under `out_root/source_name`.

    Uses content-based hashing to ensure each unique circuit block is only
    processed and rendered once, even if it appears in multiple figures.

    Returns a dict with summary information.
    """
    global _PROCESSED_BLOCKS
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

    # Track which blocks are actually new in this invocation
    blocks_saved_this_run = []

    for i, block in enumerate(blocks):
        # Create unique identifier from block content hash
        block_hash = hashlib.sha256(block.encode('utf-8')).hexdigest()[:16]
        block_id = f"{_safe_name(source_name)}_{block_hash}"
        
        # Skip if this exact block content has already been processed
        if block_id in _PROCESSED_BLOCKS:
            continue
        
        _PROCESSED_BLOCKS.add(block_id)
        
        # Use figure_number/panel if provided, otherwise fallback to old logic
        if figure_number is not None:
            fig_num = figure_number
        elif start_figure_num is not None:
            fig_num = start_figure_num + i
        else:
            fig_num = i + 1
        
        pnl = panel if panel is not None else (i + 1)
        
        # Save with unique filename that includes hash
        name = f"{_safe_name(source_name)}_fig{fig_num}p{pnl}_{block_hash}.tex"
        tex_file_path = raw_dir / name
        tex_file_path.write_text(block, encoding='utf-8')
        blocks_saved_this_run.append(tex_file_path)
        
        record['blocks'].append({
            'index': i, 
            'block_file': str((raw_dir / name).relative_to(out_root)), 
            'length': len(block),
            'block_id': block_id,
            'figure_number': fig_num,
            'panel': pnl,
            'label': figure_label
        })

        # Emit a basic circuit record if arxiv id provided
        if arxiv_id is not None:
            try:
                from core.circuit_store import emit_record
                # extract gates from the saved block
                try:
                    gates = _extract_gates_from_block(block)
                except Exception:
                    gates = []
                # prepare descriptions and text positions using provided caption_text when available
                descriptions = []
                text_positions = []
                if caption_text:
                    descriptions.append(caption_text)
                    # locate caption inside the block if possible
                    try:
                        start = block.find(caption_text)
                        if start >= 0:
                            text_positions.append((int(start), int(start + len(caption_text))))
                        else:
                            # store (0,0) as fallback
                            text_positions.append((0, 0))
                    except Exception:
                        text_positions.append((0, 0))

                rec = {
                    'arxiv_id': str(arxiv_id),
                    'page': None,
                    'figure_number': int(fig_num),
                    'panel': int(pnl),
                    'label': figure_label,
                    'gates': gates,
                    'quantum_problem': None,
                    'descriptions': descriptions,
                    'text_positions': text_positions,
                    'raw_block_file': str((raw_dir / name).as_posix()),
                    'block_id': block_id
                }
                # Defer emitting to JSONL until after successful rendering
                if 'pending_records' not in locals():
                    pending_records = []
                pending_records.append(rec)
            except Exception:
                pass

    summary_path = dest / 'summary.json'
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')

    # Only render if we actually saved new blocks in this invocation
    if render and blocks_saved_this_run:
        try:
            # Create a temporary directory with only the new blocks to render
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                tmp_raw = tmp_path / 'raw_blocks'
                tmp_raw.mkdir(parents=True, exist_ok=True)
                
                # Copy only the new blocks to temp directory
                for tex_file in blocks_saved_this_run:
                    shutil.copy2(tex_file, tmp_raw / tex_file.name)
                
                # Render only the temp directory (contains only new blocks)
                tmp_rendered = tmp_path / 'rendered'
                render_saved_blocks_with_pdflatex_module(blocks_root=str(tmp_path), out_dir=str(tmp_rendered))
                
                # Move rendered PDFs to the main rendered directory
                rendered_dir = dest / 'rendered'
                rendered_dir.mkdir(parents=True, exist_ok=True)
                
                new_pdfs = []
                if tmp_rendered.exists():
                    for pdf in tmp_rendered.glob('*.pdf'):
                        dest_pdf = rendered_dir / pdf.name
                        shutil.copy2(pdf, dest_pdf)
                        new_pdfs.append(dest_pdf)

                # Copy any pdflatex logs back to the original raw_blocks dir so logs persist
                try:
                    for log_file in tmp_path.rglob('*.log'):
                        try:
                            # copy log into the real raw_blocks directory
                            shutil.copy2(log_file, raw_dir / log_file.name)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Also copy rendered PDFs into a common folder with a paper-specific prefix
            common_dir = Path('circuit_images/rendered_pdflatex')
            common_dir.mkdir(parents=True, exist_ok=True)

            safe_src = _safe_name(source_name)
            # prepare per-paper PNG folder
            rendered_png_dir = dest / 'rendered_png'
            rendered_png_dir.mkdir(parents=True, exist_ok=True)

            # prepare common PNG folder under the shared directory
            common_png_dir = common_dir / 'png'
            common_png_dir.mkdir(parents=True, exist_ok=True)

            # Process only the newly rendered PDFs
            # Emit JSONL records only for blocks that successfully rendered
            try:
                produced_stems = {p.stem for p in new_pdfs}
            except Exception:
                produced_stems = set()

            if 'pending_records' in locals() and pending_records:
                try:
                    from core.circuit_store import emit_record
                    for rec in pending_records:
                        try:
                            rb_name = Path(rec.get('raw_block_file', '')).name
                            stem = Path(rb_name).stem
                            if stem in produced_stems:
                                emit_record(rec)
                        except Exception:
                            continue
                except Exception:
                    pass
            for f in new_pdfs:
                if f.is_file() and f.suffix.lower() == '.pdf':
                        try:
                            ts = int(time.time() * 1000)
                            # unique dest pdf name with timestamp
                            dest_pdf_name = f"{safe_src}__{f.stem}__{ts}.pdf"
                            shutil.copyfile(f, common_dir / dest_pdf_name)

                            # Convert per-paper PDF to PNG into per-paper png folder with unique name
                            try:
                                per_png_name = f"{f.stem}__{ts}.png"
                                per_png_path = rendered_png_dir / per_png_name
                                _pdf_to_png(f, per_png_path)
                            except Exception:
                                pass

                            # Convert copied common PDF to PNG into common png folder with matching unique name
                            try:
                                common_pdf = common_dir / dest_pdf_name
                                common_png_name = Path(dest_pdf_name).with_suffix('.png').name
                                common_png_path = common_png_dir / common_png_name
                                _pdf_to_png(common_pdf, common_png_path)
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


def reset_processed_blocks():
    """Reset the global set of processed blocks.
    Call this at the start of each pipeline run to start fresh.
    """
    global _PROCESSED_BLOCKS
    _PROCESSED_BLOCKS.clear()


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


#!/usr/bin/env python3
r"""
Live LaTeX extractor: process a single .tex or text input to extract
`\Qcircuit{...}` blocks and save them under `circuit_images/live_blocks/`.

This module is intentionally standalone and safe to import: it will try to
reuse the extractor from `core.latex_extractor` if available and fall back to
an embedded extractor implementation otherwise.

Usage (quick):
    python -m core.live_latex_extractor somefile.tex --render

Functions:
 - process_file(path, out_root, render, render_with_module)
 - process_text(text, source_name, out_root, render, render_with_module)
"""
from pathlib import Path
import json
import argparse
import sys

try:
    from core.latex_extractor import (
        extract_qcircuit_blocks_from_text,
        render_saved_blocks_with_pdflatex_module,
    )
except Exception:
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

    def render_saved_blocks_with_pdflatex_module(*args, **kwargs):
        raise RuntimeError('pdflatex module renderer not available')


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
            if render_with_module:
                render_saved_blocks_with_pdflatex_module(blocks_root=str(out_root), out_dir=str(out_root / 'rendered'))
            else:
                # Defer to the main module renderer if available; it may be imported above.
                try:
                    from core.latex_extractor import render_saved_blocks
                    render_saved_blocks(blocks_root=str(out_root), out_pdf_dir=str(out_root / 'rendered'))
                except Exception:
                    # fall back to module renderer
                    render_saved_blocks_with_pdflatex_module(blocks_root=str(out_root), out_dir=str(out_root / 'rendered'))
        except Exception:
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


def _main_cli():
    ap = argparse.ArgumentParser(description=r'Extract \Qcircuit blocks from a single file or STDIN')
    ap.add_argument('path', nargs='?', help='Path to .tex/.txt file. If omitted, reads stdin.')
    ap.add_argument('--out', default='circuit_images/live_blocks', help='Output root dir')
    ap.add_argument('--render', action='store_true', help='Attempt to render produced blocks to PDF')
    ap.add_argument('--render-module', action='store_true', help='Prefer Python pdflatex module renderer')
    args = ap.parse_args()

    if args.path:
        summary = process_file(args.path, out_root=args.out, render=args.render, render_with_module=args.render_module)
    else:
        text = sys.stdin.read()
        summary = process_text(text, source_name='stdin', out_root=args.out, render=args.render, render_with_module=args.render_module)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    _main_cli()

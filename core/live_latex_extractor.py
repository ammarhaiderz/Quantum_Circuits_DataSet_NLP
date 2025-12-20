#!/usr/bin/env python3
"""Extract live LaTeX Qcircuit blocks from text files and optionally render them.
"""
from pathlib import Path
import json
import argparse
import sys
import shutil
from typing import Optional
import hashlib
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
import re
import time
from tqdm import tqdm
import tempfile
from utils.latex_utils_emb import render_saved_blocks

# Global set to track processed block content hashes (prevents duplicates)
_PROCESSED_BLOCKS = set()

def render_saved_blocks_with_pdflatex_module(blocks_root: str = 'circuit_images/blocks', out_dir: str = 'circuit_images/rendered_pdflatex'):
	"""Render saved raw block files using the `pdflatex` Python wrapper (PDFLaTeX).

	Falls back to the subprocess renderer if the module is not available.
	"""
	try:
		from pdflatex import PDFLaTeX
	except Exception:
		print('pdflatex Python module not available; falling back to subprocess renderer')
		return render_saved_blocks(blocks_root, out_dir)

	blocks_root = Path(blocks_root)
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	tex_files = list(blocks_root.rglob('raw_blocks/*.tex'))
	if not tex_files:
		print(f'No raw block .tex files found under {blocks_root}')
		return

	# default wrapper uses qcircuit; per-file selection below will use quantikz if needed
	wrapper = (
		'\\documentclass[border=30pt]{standalone}\n'
		'\\usepackage{amsmath}\n'
		'\\usepackage{amssymb}\n'
		'\\usepackage{braket}\n'
		'\\usepackage{qcircuit}\n'
		"% Fallback macro definitions for extracted snippets (non-invasive)\n"
		"\\providecommand{\\psx}[1]{\\psi_{#1}}\n"
		"\\providecommand{\\psr}[1]{\\psi_{#1}}\n"
		"\\providecommand{\\pst}[1]{\\psi_{#1}}\n"
		"\\providecommand{\\rec}{\\mathrm{Rec}}\n"
		"\\providecommand{\\tra}{\\mathrm{Tr}}\n"
		"\\providecommand{\\ora}{\\mathcal{O}}\n"
		'\\begin{document}\n'
		'{circuit_code}\n'
		'\\end{document}\n'
	)

	for tex_path in tqdm(tex_files, desc='Rendering with pdflatex module'):
		try:
			block_text = tex_path.read_text(encoding='utf-8')
		except Exception:
			continue

		# Build wrapper tex content and write a temp file; choose quantikz wrapper if block uses quantikz
		lower = block_text.lower()
		if ('quantikz' in lower) or ('\\begin{quantikz' in block_text):
			content = (
				'\\documentclass[border=30pt]{standalone}\n'
				'\\usepackage{amsmath}\n'
				'\\usepackage{amssymb}\n'
				'\\usepackage{braket}\n'
				'\\usepackage{tikz}\n'
				'\\usepackage{quantikz}\n'
				'\\begin{document}\n'
				f"{block_text}\n"
				'\\end{document}\n'
			)
		else:
			content = wrapper.replace('{circuit_code}', block_text)
		# Use temp directory to compile
		with tempfile.TemporaryDirectory() as td:
			td_path = Path(td)
			tex_file = td_path / 'circuit.tex'
			tex_file.write_text(content, encoding='utf-8')

			try:
				pdfl = PDFLaTeX.from_texfile(str(tex_file))
				pdf_bytes, log_text, proc = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=True)

				# Normalize log_text to str before writing (pdflatex module may return bytes)
				if isinstance(log_text, bytes):
					try:
						log_text_str = log_text.decode('utf-8', errors='replace')
					except Exception:
						log_text_str = str(log_text)
				else:
					log_text_str = str(log_text) if log_text is not None else ''

				# Save log next to original block
				log_path = tex_path.with_suffix('.pdflatex_module.log')
				log_path.write_text(log_text_str, encoding='utf-8')

				# Ensure pdf_bytes is bytes before writing
				if proc and proc.returncode == 0 and pdf_bytes:
					out_pdf = out_dir / (tex_path.stem + '.pdf')
					if isinstance(pdf_bytes, str):
						try:
							pdf_bytes_to_write = pdf_bytes.encode('utf-8')
						except Exception:
							pdf_bytes_to_write = bytes(pdf_bytes)
					else:
						pdf_bytes_to_write = pdf_bytes
					out_pdf.write_bytes(pdf_bytes_to_write)
				else:
					# fallback: copy any produced pdf from temp dir
					produced = list(td_path.glob('*.pdf'))
					if produced:
						shutil.copyfile(produced[0], out_dir / (tex_path.stem + '.pdf'))

			except Exception as e:
				# save exception to log
				log_path = tex_path.with_suffix('.pdflatex_module.err')
				log_path.write_text(str(e), encoding='utf-8')

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
    found = []
    try:
        # determine environment
        is_qc = '\\Qcircuit' in block
        is_qk = re.search(r'\\begin\{quantikz\}', block, re.IGNORECASE) is not None

        # extract inner grid text
        inner = block
        if is_qk:
            inner = re.sub(r'\\begin\{quantikz\}(?:\[[^\]]*\])?', '', inner, flags=re.IGNORECASE)
            inner = re.sub(r'\\end\{quantikz\}', '', inner, flags=re.IGNORECASE)
        elif is_qc:
            m = re.search(r'\\Qcircuit[^\{]*\{', block)
            if m:
                start = m.end()
                depth = 1
                i = start
                while i < len(block) and depth > 0:
                    if block[i] == '{':
                        depth += 1
                    elif block[i] == '}':
                        depth -= 1
                    i += 1
                inner = block[start:i - 1]

        # split rows
        rows = [r.strip() for r in re.split(r'\\\\\s*', inner) if r.strip()]

        # parse grid
        grid = []
        maxcols = 0
        for r in rows:
            r2 = re.sub(r'%.*', '', r)
            cells = [c.strip() for c in r2.split('&')]
            grid.append(cells)
            maxcols = max(maxcols, len(cells))

        for row in grid:
            while len(row) < maxcols:
                row.append('')

        # ---------- helpers ----------

        def canonicalize_label(lbl: str) -> str:
            if not lbl:
                return ''

            s = lbl.strip()

            # unwrap wrappers
            s = re.sub(r'\\text\{([^}]*)\}', r'\1', s, flags=re.IGNORECASE)
            s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s, flags=re.IGNORECASE)
            s = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', s, flags=re.IGNORECASE)
            s = re.sub(r'\\mathcal\{([^}]*)\}', r'\1', s, flags=re.IGNORECASE)
            s = re.sub(r'\\hat\{([^}]*)\}', r'\1', s, flags=re.IGNORECASE)

            # rotations (must be BEFORE stripping punctuation)
            if re.search(r'R\s*[_\{]?\s*x\s*[\}]?\s*\(', s, re.IGNORECASE):
                return 'RX'
            if re.search(r'R\s*[_\{]?\s*y\s*[\}]?\s*\(', s, re.IGNORECASE):
                return 'RY'
            if re.search(r'R\s*[_\{]?\s*z\s*[\}]?\s*\(', s, re.IGNORECASE):
                return 'RZ'

            # dagger
            s = re.sub(r'(?:\^?\s*\\dagger|\\dagger)', 'DG', s)

            # strip latex + symbols
            s = re.sub(r'\\+', '', s)
            s = re.sub(r'[^A-Za-z0-9_]', '', s)
            s = re.sub(r'_+', '', s)

            s = re.sub(r'^TEXT', '', s)
            
            s = s.upper()
            if re.fullmatch(r'\d+', s):
                return ''
            return s

        def detect_cell(cell_text):
            tokens = []
            t = cell_text

            # qcircuit multigate
            m = re.search(r'\\multigate\{\d+\}\{([^}]*)\}', t)
            if m:
                tokens.append(('U', m.group(1).strip()))

            if re.search(r'\\ghost\{[^}]+\}', t):
                tokens.append(('GHOST', None))

            # quantikz multi-wire gate
            m = re.search(r'\\gate\s*\[([^\]]*)\]\s*\{([^}]*)\}', t)
            if m:
                params = m.group(1)
                label = m.group(2).strip()
                wm = re.search(r'wires\s*=\s*(\d+)', params)
                if wm and int(wm.group(1)) >= 2:
                    tokens.append(('U', label))
                    return tokens
                else:
                    tokens.append(('G', label))
                    return tokens

            # single gate
            m = re.search(r'\\gate\{([^}]*)\}', t)
            if m:
                tokens.append(('G', m.group(1).strip()))

            # controls
            if re.search(r'\\o?ctrl\{-?\d+\}', t) or re.search(r'\\control\b', t) or re.search(r'\\cctrl\{-?\d+\}', t):
                if re.search(r'\\octrl', t):
                    tokens.append(('O', None))
                else:
                    tokens.append(('C', None))

            # targets
            if re.search(r'\\otarg\b|\\targ\b', t):
                tokens.append(('X', None))

            # swap
            if re.search(r'\\swap\b|\\qswap\b', t):
                tokens.append(('S', None))

            # measure/reset
            if re.search(r'\\meter\b|\\measure\b', t):
                tokens.append(('M', None))
            if re.search(r'\\reset\b', t):
                tokens.append(('R', None))

            return tokens

        # ---------- collect columns ----------

        cols = [[] for _ in range(maxcols)]
        for ri, row in enumerate(grid):
            for ci, cell in enumerate(row):
                for tk, lbl in detect_cell(cell):
                    cols[ci].append((tk, lbl, ri))

        inferred = []

        for col in cols:
            if not col:
                continue

            u_labels = [lbl for tk, lbl, _ in col if tk == 'U' and lbl]
            if u_labels:
                gl = canonicalize_label(u_labels[0])
                inferred.append(gl if gl else 'U')
                continue
            elif any(tk == 'U' for tk, _, _ in col):
                inferred.append('U')
                continue

            control_count = sum(1 for tk, _, _ in col if tk in ('C', 'O'))
            x_count = sum(1 for tk, _, _ in col if tk == 'X')
            gates_in_col = [lbl for tk, lbl, _ in col if tk == 'G' and lbl]

            if any(tk == 'S' for tk, _, _ in col):
                inferred.append('SWAP')
                continue
            if any(tk == 'M' for tk, _, _ in col):
                inferred.append('MEASURE')
                continue
            if any(tk == 'R' for tk, _, _ in col):
                inferred.append('RESET')
                continue

            if x_count > 0 and control_count > 0:
                if control_count == 1:
                    inferred.append('CNOT')
                elif control_count == 2:
                    inferred.append('TOFFOLI')
                else:
                    inferred.append(f'MCX_{control_count}')
                continue

            if control_count > 0 and gates_in_col:
                gl = canonicalize_label(gates_in_col[0])
                if gl:
                    inferred.append(f'CTRL-{gl}' if control_count == 1 else f'MCTRL{control_count}-{gl}')
                else:
                    inferred.append('CTRL')
                continue

            for g in gates_in_col:
                gl = canonicalize_label(g)
                if gl:
                    inferred.append('H' if gl in ('HADAMARD', 'HAT') else gl)

            if x_count > 0 and control_count == 0:
                inferred.append('X')

        if not inferred:
            for txt in re.findall(r"\b(H|X|Y|Z|CNOT|CX|SWAP|TOFFOLI|CCX|S|T|RZ|RX|RY)\b", block, re.IGNORECASE):
                inferred.append(txt.upper())

        out = []
        seen = set()
        for g in inferred:
            g2 = g.upper()
            if g2 and g2 not in seen:
                seen.add(g2)
                out.append(g2)
        return out

    except Exception:
        try:
            out = []
            seen = set()
            for txt in re.findall(r"\b(H|X|Y|Z|CNOT|CX|SWAP|TOFFOLI|CCX|S|T|RZ|RX|RY)\b", block, re.IGNORECASE):
                t = txt.upper()
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out
        except Exception:
            return []


def process_text(text: str, source_name: str = 'inline', out_root: str = 'circuit_images/live_blocks', render: bool = True, render_with_module: bool = False, arxiv_id: Optional[str] = None, start_figure_num: Optional[int] = None, caption_text: Optional[str] = None, panel: Optional[int] = None, figure_label: Optional[str] = None):
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
    # blocks to skip rendering because they have no captions/descriptions
    _skip_render_blocks = set()

    for i, block in enumerate(blocks):
        # Create unique identifier from block content hash
        block_hash = hashlib.sha256(block.encode('utf-8')).hexdigest()[:16]
        block_id = f"{_safe_name(source_name)}_{block_hash}"
        
        # Skip if this exact block content has already been processed
        if block_id in _PROCESSED_BLOCKS:
            continue
        
        _PROCESSED_BLOCKS.add(block_id)
        
        # Determine panel index; do not use figure numbering
        pnl = panel if panel is not None else (i + 1)
        # (NO per-block quota check here; rendering-level quota is enforced later)

        # Save with unique filename that includes hash (no figure number)
        name = f"{_safe_name(source_name)}_p{pnl}_{block_hash}.tex"
        tex_file_path = raw_dir / name
        tex_file_path.write_text(block, encoding='utf-8')
        blocks_saved_this_run.append(tex_file_path)
        
        record['blocks'].append({
            'index': i,
            'block_file': str((raw_dir / name).relative_to(out_root)),
            'length': len(block),
            'block_id': block_id,
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
                    'label': figure_label,
                    'gates': gates,
                    'quantum_problem': None,
                    'descriptions': descriptions,
                    'text_positions': text_positions
                }
                # Defer emitting to JSONL until after successful rendering
                if 'pending_records' not in locals():
                    pending_records = []
                # If there are no descriptions (captions), skip creating a pending record
                # and mark this block to be skipped for rendering so the rendered folder
                # stays consistent with emitted records.
                if not descriptions:
                    _skip_render_blocks.add(tex_file_path.name)
                else:
                    # store pending record together with its tex filename (not persisted)
                    pending_records.append((rec, name))
            except Exception:
                pass

    summary_path = dest / 'summary.json'
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')

    # Only render if we actually saved new blocks in this invocation
    if render and blocks_saved_this_run:
        try:
            # If the canonical circuits JSON already has >=250 items, skip
            # further circuit rendering/emission to let the main image
            # extraction continue independently.
            try:
                from core.circuit_store import JSON_PATH
                import json as _json
                if JSON_PATH.exists():
                    try:
                        with open(JSON_PATH, 'r', encoding='utf-8') as _jf:
                            _data = _json.load(_jf)
                            if isinstance(_data, dict) and len(_data) >= 250:
                                # skip rendering/emitting records
                                return record
                    except Exception:
                        # if JSON can't be read, fall back to rendering
                        pass
            except Exception:
                # circuit_store not available; continue as before
                pass
            # Create a temporary directory with only the new blocks to render
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                tmp_raw = tmp_path / 'raw_blocks'
                tmp_raw.mkdir(parents=True, exist_ok=True)
                
                # Copy only the new blocks to temp directory
                for tex_file in blocks_saved_this_run:
                    # skip blocks that were marked as having no captions
                    if tex_file.name in _skip_render_blocks:
                        continue
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

            # Generate per-paper and common PNGs for the new PDFs first,
            # building a mapping from pdf stem -> common PNG filename.
            # Enforce rendering quota: compute how many images we are allowed to
            # create (remaining = MAX_IMAGES - current_count). Only process up to
            # that many PDFs from `new_pdfs` and emit records only for actually
            # produced PNGs.
            stem_to_common_png = {}
            try:
                from config.settings import MAX_IMAGES
                from core.circuit_store import JSON_PATH
                import json as _json
                current = 0
                if JSON_PATH.exists():
                    try:
                        with open(JSON_PATH, 'r', encoding='utf-8') as _jf:
                            _d = _json.load(_jf)
                            if isinstance(_d, dict):
                                current = len(_d)
                    except Exception:
                        current = 0
                remaining = MAX_IMAGES - current
                if remaining <= 0:
                    # no capacity to render more images
                    new_pdfs_to_process = []
                else:
                    new_pdfs_to_process = new_pdfs[:remaining]
            except Exception:
                # fallback: process all
                new_pdfs_to_process = new_pdfs

            try:
                for f in new_pdfs_to_process:
                    if not (f.is_file() and f.suffix.lower() == '.pdf'):
                        continue
                    try:
                        ts = int(time.time() * 1000)
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
                            stem_to_common_png[f.stem] = common_png_name
                        except Exception:
                            pass
                    except Exception:
                        # Non-fatal; continue copying other files
                        pass
            except Exception:
                stem_to_common_png = {}

            # Emit JSONL records only for blocks that successfully rendered
            if 'pending_records' in locals() and pending_records:
                try:
                    from core.circuit_store import emit_record
                    for rec, bn in pending_records:
                        try:
                            stem = Path(bn).stem
                            png_name = stem_to_common_png.get(stem)
                            if png_name:
                                # Attach the produced common PNG filename so the store
                                # can reliably map this JSONL record to the image.
                                try:
                                    rec['image_filename'] = png_name
                                except Exception:
                                    pass
                                # Remove legacy auxiliary fields but keep `image_filename`
                                for _f in ('raw_block_file', 'block_id', 'block_name'):
                                    try:
                                        rec.pop(_f, None)
                                    except Exception:
                                        pass
                                emit_record(rec)
                        except Exception:
                            continue
                except Exception:
                    pass

            # # After emitting records, if we've reached the configured MAX_IMAGES,
            # # remove any stray PNGs in the common PNG folder that are not keys
            # # in the canonical `data/circuits.json` to avoid extra images.
            # try:
            #     from config.settings import MAX_IMAGES
            #     from tools.cleanup_extra_pngs import cleanup_extra_pngs
            #     from core.circuit_store import JSON_PATH
            #     if JSON_PATH.exists():
            #         import json as _json
            #         with open(JSON_PATH, 'r', encoding='utf-8') as _jf:
            #             _data = _json.load(_jf)
            #         if isinstance(_data, dict) and len(_data) >= MAX_IMAGES:
            #             try:
            #                 cleanup_extra_pngs(str(JSON_PATH), str(common_png_dir), max_images=MAX_IMAGES)
            #             except Exception:
            #                 pass
            # except Exception:
            #     pass

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


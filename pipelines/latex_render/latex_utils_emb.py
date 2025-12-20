import re
import tarfile
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm

from config.settings import LATEX_BLOCKS_ROOT, LATEX_RENDER_DIR


def render_saved_blocks(blocks_root: str = None, out_pdf_dir: str = None):
	"""Render saved raw block files to PDF using system ``pdflatex``.

	Looks for ``*/raw_blocks/*.tex`` under ``blocks_root``, wraps each block into a
	standalone LaTeX document, runs ``pdflatex`` twice, saves ``.log`` files, and
	copies resulting PDFs to ``out_pdf_dir``.

	Parameters
	----------
	blocks_root : str, optional
		Root directory containing ``raw_blocks`` subfolders (default
		``'circuit_images/blocks'``).
	out_pdf_dir : str, optional
		Destination directory for rendered PDFs (default
		``'circuit_images/rendered'``).
	"""
	blocks_root = Path(blocks_root or LATEX_BLOCKS_ROOT)
	out_pdf_dir = Path(out_pdf_dir or LATEX_RENDER_DIR)
	out_pdf_dir.mkdir(parents=True, exist_ok=True)

	# Default wrapper uses qcircuit; quantikz-only wrappers are chosen per-file
	latex_template = (
		"\\documentclass[border=30pt]{standalone}\n"
		"\\usepackage{amsmath}\n"
		"\\usepackage{amssymb}\n"
		"\\usepackage{braket}\n"
		"\\usepackage{qcircuit}\n"
		"% Fallback macro definitions for extracted snippets (non-invasive)\n"
		"\\providecommand{\\psx}[1]{\\psi_{#1}}\n"
		"\\providecommand{\\psr}[1]{\\psi_{#1}}\n"
		"\\providecommand{\\pst}[1]{\\psi_{#1}}\n"
		"\\providecommand{\\rec}{\\mathrm{Rec}}\n"
		"\\providecommand{\\tra}{\\mathrm{Tr}}\n"
		"\\providecommand{\\ora}{\\mathcal{O}}\n"

		"\\begin{document}\n"
		"{circuit_code}\n"
		"\\end{document}\n"
	)

	tex_files = list(blocks_root.rglob('raw_blocks/*.tex'))
	if not tex_files:
		print(f"No raw block .tex files found under {blocks_root}")
		return

	print(f"Rendering {len(tex_files)} blocks to PDF...")

	for tex_path in tqdm(tex_files, desc='Rendering blocks'):
		try:
			block_text = tex_path.read_text(encoding='utf-8')
		except Exception:
			continue

		# Choose wrapper: if this block appears to use quantikz, use a quantikz-friendly wrapper
		lower = block_text.lower()
		if ('quantikz' in lower) or ('\\begin{quantikz' in block_text):
			quantikz_template = (
				"\\documentclass[border=30pt]{standalone}\n"
				"\\usepackage{amsmath}\n"
				"\\usepackage{amssymb}\n"
				"\\usepackage{braket}\n"
				"\\usepackage{tikz}\n"
				"\\usepackage{quantikz}\n"
				"% Fallback macro definitions for extracted snippets (non-invasive)\n"
				"\\providecommand{\\psx}[1]{\\psi_{#1}}\n"
				"\\providecommand{\\psr}[1]{\\psi_{#1}}\n"
				"\\providecommand{\\pst}[1]{\\psi_{#1}}\n"
				"\\providecommand{\\rec}{\\mathrm{Rec}}\n"
				"\\providecommand{\\tra}{\\mathrm{Tr}}\n"
				"\\providecommand{\\ora}{\\mathcal{O}}\n"
				"\\begin{document}\n"
				"{circuit_code}\n"
				"\\end{document}\n"
			)
			wrapper_tex = quantikz_template.replace('{circuit_code}', block_text)
		else:
			wrapper_tex = latex_template.replace('{circuit_code}', block_text)

		# Use a temporary directory to compile to avoid name collisions
		with tempfile.TemporaryDirectory() as td:
			td_path = Path(td)
			wrap_name = tex_path.stem + '_wrap.tex'
			wrap_path = td_path / wrap_name
			wrap_path.write_text(wrapper_tex, encoding='utf-8')

			# Run pdflatex twice
			# invoke pdflatex from the temp directory and pass the filename (avoids issues with spaces in full path)
			cmd = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', wrap_path.name]
			log_text = ''
			success = False
			for _ in range(2):
				try:
					res = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=45)
					log_text += res.stdout + '\n' + res.stderr + '\n'
					if res.returncode != 0:
						success = False
						break
					else:
						success = True
				except subprocess.TimeoutExpired:
					log_text += 'pdflatex timeout\n'
					success = False
					break

			# Save log next to original tex block for debugging
			log_dir = tex_path.parent
			log_dir.mkdir(parents=True, exist_ok=True)
			log_file = log_dir / (tex_path.stem + '.pdflatex.log')
			log_file.write_text(log_text, encoding='utf-8')

			if success:
				produced_pdf = td_path / (wrap_path.stem + '.pdf')
				if produced_pdf.exists():
					dest_pdf = out_pdf_dir / (tex_path.stem + '.pdf')
					shutil.copyfile(produced_pdf, dest_pdf)
				else:
					found = list(td_path.glob('*.pdf'))
					if found:
						shutil.copyfile(found[0], out_pdf_dir / (tex_path.stem + '.pdf'))

	print('Rendering complete. PDFs in', out_pdf_dir)
def render_saved_blocks_with_pdflatex_module(blocks_root: str = None, out_dir: str = None):
	"""Render saved raw block files using the ``pdflatex`` Python wrapper.

	Falls back to the subprocess renderer when the module is unavailable.

	Parameters
	----------
	blocks_root : str, optional
		Root directory containing ``raw_blocks`` subfolders (default
		``'circuit_images/blocks'``).
	out_dir : str, optional
		Destination directory for rendered PDFs (default
		``'circuit_images/rendered_pdflatex'``).
	"""
	try:
		from pdflatex import PDFLaTeX
	except Exception:
		print('pdflatex Python module not available; falling back to subprocess renderer')
		return render_saved_blocks(blocks_root, out_dir)

	blocks_root = Path(blocks_root or LATEX_BLOCKS_ROOT)
	out_dir = Path(out_dir or LATEX_RENDER_DIR)
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
	r"""Extract complete ``\Qcircuit{...}`` blocks from text.

	Strategy:
	- Find occurrences of optional ``\label{...}`` followed by ``\Qcircuit``.
	- Locate the first ``{`` after the match and perform a balanced-brace scan
	  to extract the full block.
	- Return a list of blocks (strings).

	Parameters
	----------
	text : str
		LaTeX source to scan.

	Returns
	-------
	list[str]
		Unique Qcircuit blocks found in the text.
	"""

	blocks = []

	def _line_is_commented(txt: str, idx: int) -> bool:
		"""Return True if the position `idx` lies on a line that is commented out
		(i.e. there is an unescaped `%` between the start of the line and idx).
		"""
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

	# Pattern matches optional label then \Qcircuit
	start_pattern = re.compile(r'(?:\\label\{[^}]*\}\s*)?\\Qcircuit\b', re.IGNORECASE)

	for m in start_pattern.finditer(text):
		if _line_is_commented(text, m.start()):
			continue
		start_idx = m.start()

		# find first '{' after the match
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
					# simple validation: must contain a circuit token
					if re.search(r'\\(qw|gate|ctrl|targ|meter|lstick|rstick|cw)\b', block):
						blocks.append(block)
					break
			i += 1

	# Deduplicate, keep order
	seen = set()
	unique = []
	for b in blocks:
		key = b.strip()[:300]
		if key not in seen:
			seen.add(key)
			unique.append(b)

	return unique


def safe_name(name: str) -> str:
	"""Return a filesystem-safe version of a name."""
	return name.replace('/', '__').replace('..', '__')


def process_all_tars(tar_folder: str = 'arxiv_cache', out_root: str = 'circuit_images/blocks'):
	"""Process all tar archives and extract/render Qcircuit blocks.

	Parameters
	----------
	tar_folder : str, optional
		Directory containing ``.tar`` or ``.tar.gz`` archives (default ``'arxiv_cache'``).
	out_root : str, optional
		Root output directory for extracted blocks and summaries (default
		``'circuit_images/blocks'``).
	"""
	tar_dir = Path(tar_folder)
	out_root = Path(out_root)
	out_root.mkdir(parents=True, exist_ok=True)

	if not tar_dir.exists():
		print(f"Tar folder not found: {tar_dir}")
		return

	# Collect tar-like files
	tar_files = sorted([p for p in tar_dir.glob('*.tar*') if p.is_file()])
	if not tar_files:
		print(f"No tar files found in {tar_dir}")
		return

	total_blocks = 0
	total_papers_with_blocks = 0

	for tar_path in tar_files:
		tar_stem = tar_path.stem
		print(f"Processing archive: {tar_path.name}")

		try:
			with tarfile.open(tar_path, 'r:*') as tar:
				tex_members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith('.tex')]

				for member in tqdm(tex_members, desc=f"Files in {tar_path.name}"):
					try:
						f = tar.extractfile(member)
						if f is None:
							continue
						raw = f.read()
						try:
							text = raw.decode('utf-8')
						except UnicodeDecodeError:
							text = raw.decode('latin-1', errors='ignore')

						blocks = extract_qcircuit_blocks_from_text(text)
						if not blocks:
							continue

						total_papers_with_blocks += 1

						# Prepare output paths
						subdir = out_root / tar_stem
						subdir.mkdir(parents=True, exist_ok=True)

						member_safe = safe_name(member.name)
						json_path = subdir / f"{member_safe}.json"

						paper_record = {
							'tar_file': tar_path.name,
							'tex_file': member.name,
							'blocks_count': len(blocks),
							'blocks': []
						}

						# Write blocks into a member-specific folder to avoid re-rendering other papers
						raw_blocks_dir = subdir / member_safe / 'raw_blocks'
						raw_blocks_dir.mkdir(parents=True, exist_ok=True)

						for i, block in enumerate(blocks):
							block_name = f"{member_safe}_block_{i:03d}.tex"
							(raw_blocks_dir / block_name).write_text(block, encoding='utf-8')
							paper_record['blocks'].append({
								'index': i,
								'preview': block[:300] + ('...' if len(block) > 300 else ''),
								'length': len(block),
								'block_file': str((raw_blocks_dir / block_name).relative_to(out_root))
							})

						# Save JSON summary for this tex file
						with open(json_path, 'w', encoding='utf-8') as jf:
							json.dump(paper_record, jf, indent=2, ensure_ascii=False)

						total_blocks += len(blocks)

						# Render blocks for this paper only (member-specific folder)
						try:
							render_saved_blocks_with_pdflatex_module(blocks_root=str(subdir / member_safe), out_dir='circuit_images/rendered_pdflatex')
						except Exception:
							# don't fail the extraction process if rendering fails
							pass

					except Exception as e:
						# continue on errors per file
						continue

		except Exception as e:
			print(f"Failed to open {tar_path}: {e}")
			continue

	print("\nExtraction complete")
	print(f"Papers with blocks: {total_papers_with_blocks}")
	print(f"Total blocks saved: {total_blocks}")


"""
Image extraction utilities from arXiv sources.
"""

import os
import re
import tarfile
import io
from typing import List, Set, Optional
import requests
import time

from shared.figure_data import Figure, ExtractedImage
from shared.preprocessor import TextPreprocessor
from config.settings import (
    SUPPORTED_EXT,
    REQUEST_DELAY,
    IMAGE_PIPELINE_OUTPUT_DIR,
    IMAGE_PIPELINE_CACHE_DIR,
    IMAGE_PIPELINE_PDF_CACHE_DIR,
)
from config.queries import FILENAME_NEGATIVE_TOKENS


class ImageExtractor:
    """Handles extraction of images from arXiv sources."""
    
    # Regular expressions for LaTeX parsing
    FIG_RE = re.compile(r"\\begin{figure}.*?\\end{figure}", re.DOTALL)
    CAP_RE = re.compile(r"\\caption\{([^}]*)\}", re.DOTALL)
    LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
    SUBCAP_RE = re.compile(r"\\subcaption\{([^}]*)\}", re.DOTALL)
    IMG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", re.DOTALL)
    SUBFIG_RE = re.compile(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", re.DOTALL)
    # LaTeX drawing environments/macros commonly used for circuits
    TIKZ_RE = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.DOTALL)
    CIRCUITIKZ_RE = re.compile(r"\\begin\{circuitikz\}.*?\\end\{circuitikz\}", re.DOTALL)
    QUANTIKZ_RE = re.compile(r"\\begin\{quantikz\}.*?\\end\{quantikz\}", re.DOTALL)
    QCIRCUIT_RE = re.compile(r"\\Qcircuit[\s\S]*?\\end{array}", re.DOTALL)
    
    def __init__(self, preprocessor: TextPreprocessor):
        """Initialize the extractor with preprocessing utilities.

        Parameters
        ----------
        preprocessor : TextPreprocessor
            Text preprocessor used for filename and caption handling.
        """
        self.preprocessor = preprocessor
        self.saved_images: Set[str] = set()
        # Tracks live-render figure numbering across multiple tex files
        self.figure_counter: int = 1
    
    def clear_output_dir(self, extensions: Optional[List[str]] = None):
        """Clear previous images from output directory.

        Parameters
        ----------
        extensions : list of str, optional
            File extensions to remove. Defaults to SUPPORTED_EXT when None.
        """
        if extensions is None:
            extensions = SUPPORTED_EXT
        
        for fname in os.listdir(IMAGE_PIPELINE_OUTPUT_DIR):
            fpath = os.path.join(IMAGE_PIPELINE_OUTPUT_DIR, fname)
            if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in extensions):
                os.remove(fpath)
    
    def read_arxiv_ids(self, filename: str) -> List[str]:
        """Read arXiv identifiers from a file.

        Parameters
        ----------
        filename : str
            Path to a file containing one arXiv identifier per line.

        Returns
        -------
        list of str
            List of normalized arXiv identifiers without the ``arXiv:`` prefix.
        """
        with open(filename, "r") as f:
            return [l.strip().replace("arXiv:", "") for l in f if l.strip()]
    
    def download_source(self, arxiv_id: str) -> Optional[io.BytesIO]:
        """Download or load cached arXiv source tarball.

        Parameters
        ----------
        arxiv_id : str
            Identifier of the arXiv paper (e.g., ``2301.01234``).

        Returns
        -------
        BytesIO or None
            In-memory tarball bytes when available; otherwise ``None``.
        """
        cache_path = os.path.join(IMAGE_PIPELINE_CACHE_DIR, f"{arxiv_id}.tar.gz")
        
        # Use cached version if available
        if os.path.exists(cache_path):
            print(f"[INFO] Using cached source for {arxiv_id}")
            with open(cache_path, "rb") as f:
                return io.BytesIO(f.read())
        
        # Download with delay
        print(f"\n📥 Downloading {arxiv_id}")
        time.sleep(REQUEST_DELAY)
        
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(cache_path, "wb") as f:
                    f.write(r.content)
                return io.BytesIO(r.content)
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
        
        return None
    
    def download_pdf_paper(self, arxiv_id: str) -> Optional[bytes]:
        """Download or load cached PDF paper file.

        Parameters
        ----------
        arxiv_id : str
            Identifier of the arXiv paper (e.g., ``2301.01234``).

        Returns
        -------
        bytes or None
            PDF bytes when available; otherwise ``None``.
        """
        cache_path = os.path.join(IMAGE_PIPELINE_PDF_CACHE_DIR, f"{arxiv_id}.pdf")
        
        # Use cached version if available
        if os.path.exists(cache_path):
            print(f"[INFO] Using cached PDF for {arxiv_id}")
            try:
                with open(cache_path, "rb") as f:
                    return f.read()
            except Exception as e:
                print(f"[WARN] Failed to load cached PDF: {e}")
                return None
        
        # Download with delay
        print(f"📥 Downloading PDF for {arxiv_id}")
        time.sleep(REQUEST_DELAY)
        
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                # Create cache directory
                os.makedirs(PDF_CACHE_DIR, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(r.content)
                return r.content
        except Exception as e:
            print(f"[WARN] PDF download failed: {e}")
        
        return None
    
    def get_pdf_cache_path(self, paper_id: str, filename: str) -> str:
        """Build cache path for an extracted image file.

        Parameters
        ----------
        paper_id : str
            Identifier of the arXiv paper associated with the image.
        filename : str
            Original filename of the image within the source archive.

        Returns
        -------
        str
            Absolute path where the cached image should be stored.
        """
        safe_pid = paper_id.replace("/", "_").replace(".", "_")
        cache_subdir = os.path.join(IMAGE_PIPELINE_PDF_CACHE_DIR, safe_pid)
        os.makedirs(cache_subdir, exist_ok=True)
        return os.path.join(cache_subdir, filename)
    
    def load_cached_pdf(self, paper_id: str, filename: str) -> Optional[bytes]:
        """Load an extracted image from cache if present.

        Parameters
        ----------
        paper_id : str
            Identifier of the arXiv paper.
        filename : str
            Cached filename to load.

        Returns
        -------
        bytes or None
            Cached file contents when found; otherwise ``None``.
        """
        cache_path = self.get_pdf_cache_path(paper_id, filename)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return f.read()
            except Exception as e:
                print(f"[WARN] Failed to load cached image {filename}: {e}")
        return None
    
    def save_pdf_to_cache(self, paper_id: str, filename: str, data: bytes) -> bool:
        """Save extracted image bytes to cache.

        Parameters
        ----------
        paper_id : str
            Identifier of the arXiv paper.
        filename : str
            Target filename within the cache.
        data : bytes
            Image bytes to persist.

        Returns
        -------
        bool
            ``True`` when the file is written successfully; otherwise ``False``.
        """
        cache_path = self.get_pdf_cache_path(paper_id, filename)
        try:
            with open(cache_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"[WARN] Failed to cache image {filename}: {e}")
            return False
    
    def filename_is_negative(self, img_path: str) -> bool:
        """Check whether a filename contains negative tokens.

        Parameters
        ----------
        img_path : str
            Path or name of the image file.

        Returns
        -------
        bool
            ``True`` when any token matches ``FILENAME_NEGATIVE_TOKENS``; otherwise ``False``.
        """
        fname = os.path.basename(img_path)
        tokens = self.preprocessor.preprocess_filename(fname)
        return any(t in FILENAME_NEGATIVE_TOKENS for t in tokens)

    def _extract_caption_from_block(self, block_text: str) -> Optional[str]:
        """Extract a caption from a LaTeX figure block.

        Parameters
        ----------
        block_text : str
            LaTeX snippet corresponding to a single figure environment.

        Returns
        -------
        str or None
            Raw caption contents when found; otherwise ``None``.
        """
        key = '\\caption'
        i = block_text.find(key)
        if i == -1:
            return None

        # advance past '\\caption'
        j = i + len(key)
        # skip whitespace
        while j < len(block_text) and block_text[j].isspace():
            j += 1

        # handle optional short-form [..]
        if j < len(block_text) and block_text[j] == '[':
            # find matching closing ]
            k = j + 1
            depth = 1
            while k < len(block_text) and depth > 0:
                if block_text[k] == ']' and depth == 1:
                    depth = 0
                    k += 1
                    break
                k += 1
            j = k
            # skip whitespace to '{'
            while j < len(block_text) and block_text[j].isspace():
                j += 1

        # now expect a '{'
        if j >= len(block_text) or block_text[j] != '{':
            # malformed caption
            return None

        # balanced-brace scan
        start = j + 1
        depth = 1
        k = start
        while k < len(block_text) and depth > 0:
            ch = block_text[k]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return block_text[start:k]
            k += 1
        return None
    
    def extract_figures_from_tex(self, tex: str, paper_id: Optional[str] = None, figure_counter: Optional[int] = None) -> List[Figure]:
        r"""Extract figures from LaTeX text.
        Handles multiple \includegraphics within a single figure block and subfigures.

        Parameters
        ----------
        tex : str
            Raw LaTeX source text to scan for figure environments.
        paper_id : str, optional
            Identifier of the paper, propagated to live extractor for labeling.
        figure_counter : int, optional
            Starting counter used by the live extractor for numbering blocks. When
            omitted, the internal ``figure_counter`` is used and updated.

        Returns
        -------
        list[Figure]
            Parsed figures extracted from the LaTeX source.
        """
        figures: List[Figure] = []
        counter = figure_counter if figure_counter is not None else (self.figure_counter or 1)

        for block in self.FIG_RE.findall(tex):
            raw_cap = self._extract_caption_from_block(block)
            if not raw_cap:
                # No caption; skip to reduce false positives
                continue

            # Decode caption text safely using pylatexenc when available
            try:
                from pylatexenc.latex2text import LatexNodes2Text
                caption_text = LatexNodes2Text().latex_to_text(raw_cap)
            except Exception:
                caption_text = raw_cap

            caption_text = caption_text.strip()

            subfig_blocks = self.SUBFIG_RE.findall(block)

            # Collect all image paths within the figure block
            img_paths = self.IMG_RE.findall(block) or []

            # If subfigures exist, also search within subfigure blocks explicitly
            for sub in subfig_blocks:
                img_paths.extend(self.IMG_RE.findall(sub))

            # Attempt to extract subcaptions if present
            subcaptions = []
            # Collect explicit \subcaption{...}
            for sub in subfig_blocks:
                subs = self.SUBCAP_RE.findall(sub)
                subcaptions.extend([s.strip() for s in subs if s.strip()])

            # If no \subcaption, try to split main caption by panel markers (a), (b), (c)...
            if not subcaptions:
                # Patterns like "a)" or "(a)" often separate panels in the raw caption
                # We split on occurrences of (a)/(b)/... while keeping text order.
                parts = re.split(r"\s*\(?([a-zA-Z])\)\s*", caption_text)
                # re.split returns separators; rebuild chunks labeled by a,b,c...
                labeled = []
                i = 0
                while i < len(parts):
                    if i + 1 < len(parts) and len(parts[i+1]) == 1:
                        label = parts[i+1].lower()
                        text_chunk = parts[i].strip()
                        if text_chunk:
                            labeled.append(text_chunk)
                        i += 2
                    else:
                        # Leading or trailing text without label
                        chunk = parts[i].strip()
                        if chunk:
                            labeled.append(chunk)
                        i += 1
                # If we got multiple chunks, treat them as subcaptions in order
                if len(labeled) > 1:
                    subcaptions = labeled

            # Deduplicate while preserving order
            seen = set()
            unique_paths = []
            for p in img_paths:
                p = p.strip()
                if p and p not in seen:
                    seen.add(p)
                    unique_paths.append(p)

            # Identify LaTeX-only circuit blocks
            latex_blocks = []
            latex_blocks.extend(self.TIKZ_RE.findall(block))
            latex_blocks.extend(self.CIRCUITIKZ_RE.findall(block))
            latex_blocks.extend(self.QUANTIKZ_RE.findall(block))
            latex_blocks.extend(self.QCIRCUIT_RE.findall(block))

            # Create a Figure per bitmap image. If we have subcaptions and counts match, align them; else use shared caption.
            if unique_paths:
                if subcaptions and len(subcaptions) == len(unique_paths):
                    for path, subcap in zip(unique_paths, subcaptions):
                        figures.append(Figure(
                            caption=subcap,
                            img_path=path
                        ))
                else:
                    for path in unique_paths:
                        figures.append(Figure(
                            caption=caption_text,
                            img_path=path
                        ))

            # Create placeholder Figures for LaTeX-rendered blocks when no bitmap paths are present.
            if latex_blocks and not unique_paths:
                if subcaptions and len(subcaptions) == len(latex_blocks):
                    for lb, subcap in zip(latex_blocks, subcaptions):
                        figures.append(Figure(
                            caption=subcap,
                            img_path="__LATEX_RENDER__",
                            latex_block=lb
                        ))
                else:
                    for lb in latex_blocks:
                        figures.append(Figure(
                            caption=caption_text,
                            img_path="__LATEX_RENDER__",
                            latex_block=lb
                        ))

            # If this figure block contains a \Qcircuit or quantikz, save the contained blocks
            # using the live extractor, passing the figure caption so records
            # include descriptions/text positions. Increment figure_counter by
            # the number of blocks found in this figure.
            if ('\\Qcircuit' in block or self.QCIRCUIT_RE.search(block) or 
                'quantikz' in block or self.QUANTIKZ_RE.search(block)):
                try:
                    source_name = paper_id
                    # attempt to extract a \label{...} inside the figure block
                    lab_m = self.LABEL_RE.search(block)
                    figure_label = lab_m.group(1).strip() if lab_m else None
                    from pipelines.latex_render.live_latex_extractor import process_text
                    process_text(block, source_name=source_name, render=True, render_with_module=True, arxiv_id=paper_id, start_figure_num=counter, caption_text=caption_text, figure_label=figure_label)
                    try:
                        from pipelines.latex_render.live_latex_extractor import extract_qcircuit_blocks_from_text
                        n = len(extract_qcircuit_blocks_from_text(block))
                    except Exception:
                        n = 0
                    counter = (counter or 1) + n
                except Exception:
                    pass

        self.figure_counter = counter
        return figures



    def extract_images(self, tar: tarfile.TarFile, figures: List[Figure], 
                      paper_id: str) -> List[ExtractedImage]:
        """Extract images from a tar archive for selected figures.

        Parameters
        ----------
        tar : tarfile.TarFile
            Open tarball containing the arXiv source files.
        figures : list of Figure
            Figures that were selected for extraction.
        paper_id : str
            Identifier of the arXiv paper.

        Returns
        -------
        list of ExtractedImage
            List of extracted image records with file paths and metadata.
        """
        members = {m.name: m for m in tar.getmembers()}
        extracted = []
        safe_pid = paper_id.replace("/", "_").replace(".", "_")
        
        for idx, f in enumerate(figures):
            base = f.img_path
            
            # Skip if filename is negative
            if self.filename_is_negative(base):
                continue
            
            # Try different extensions
            for ext in [""] + SUPPORTED_EXT:
                candidate = base + ext
                if candidate in members:
                    key = f"{paper_id}:{candidate}"
                    
                    # Skip if already saved
                    if key in self.saved_images:
                        continue
                    
                    # Extract image
                    try:
                        data = tar.extractfile(members[candidate]).read()
                        fname = f"{safe_pid}_{idx}_{os.path.basename(candidate)}"
                        out_path = os.path.join(IMAGE_PIPELINE_OUTPUT_DIR, fname)
                        
                        with open(out_path, "wb") as w:
                            w.write(data)
                        
                        self.saved_images.add(key)
                        
                        extracted.append(ExtractedImage(
                            file_path=out_path,
                            img_name=os.path.basename(candidate),
                            caption=f.caption,
                            preprocessed_text=f.preprocessed_text,
                            similarity=f.similarity,
                            sbert_sim=f.sbert_sim,
                            best_sbert_query=f.best_sbert_query,
                            paper_id=paper_id
                        ))
                        
                        f.extracted = True
                        
                    except Exception as e:
                        print(f"[WARN] Failed to extract {candidate}: {e}")
                    
                    break  # Stop trying extensions once found
        
        return extracted

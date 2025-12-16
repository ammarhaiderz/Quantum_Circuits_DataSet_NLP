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

from models.figure_data import Figure, ExtractedImage
from core.preprocessor import TextPreprocessor
from config.settings import SUPPORTED_EXT, REQUEST_DELAY, OUTPUT_DIR, CACHE_DIR, PDF_CACHE_DIR, ENABLE_PDF_EXTRACTION
from config.queries import FILENAME_NEGATIVE_TOKENS

# Optional live LaTeX extractor (single-file/text processor)
try:
    from core.live_latex_extractor import process_text as _live_process_text
except Exception:
    _live_process_text = None


class ImageExtractor:
    """Handles extraction of images from arXiv sources."""
    
    # Regular expressions for LaTeX parsing
    FIG_RE = re.compile(r"\\begin{figure}.*?\\end{figure}", re.DOTALL)
    CAP_RE = re.compile(r"\\caption\{([^}]*)\}", re.DOTALL)
    SUBCAP_RE = re.compile(r"\\subcaption\{([^}]*)\}", re.DOTALL)
    IMG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", re.DOTALL)
    SUBFIG_RE = re.compile(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", re.DOTALL)
    # LaTeX drawing environments/macros commonly used for circuits
    TIKZ_RE = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.DOTALL)
    CIRCUITIKZ_RE = re.compile(r"\\begin\{circuitikz\}.*?\\end\{circuitikz\}", re.DOTALL)
    QUANTIKZ_RE = re.compile(r"\\begin\{quantikz\}.*?\\end\{quantikz\}", re.DOTALL)
    QCIRCUIT_RE = re.compile(r"\\Qcircuit[\s\S]*?\\end{array}", re.DOTALL)
    
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.saved_images: Set[str] = set()
    
    def clear_output_dir(self, extensions: Optional[List[str]] = None):
        """Clear previous images from output directory."""
        if extensions is None:
            extensions = SUPPORTED_EXT
        
        for fname in os.listdir(OUTPUT_DIR):
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in extensions):
                os.remove(fpath)
    
    def read_arxiv_ids(self, filename: str) -> List[str]:
        """Read arXiv IDs from file."""
        with open(filename, "r") as f:
            return [l.strip().replace("arXiv:", "") for l in f if l.strip()]
    
    def download_source(self, arxiv_id: str) -> Optional[io.BytesIO]:
        """Download or load cached arXiv source."""
        cache_path = os.path.join(CACHE_DIR, f"{arxiv_id}.tar.gz")
        
        # Use cached version if available
        if os.path.exists(cache_path):
            print(f"📦 Using cached source for {arxiv_id}")
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
            print(f"❌ Download failed: {e}")
        
        return None
    
    def download_pdf_paper(self, arxiv_id: str) -> Optional[bytes]:
        """Download or load cached PDF paper file."""
        cache_path = os.path.join(PDF_CACHE_DIR, f"{arxiv_id}.pdf")
        
        # Use cached version if available
        if os.path.exists(cache_path):
            print(f"📄 Using cached PDF for {arxiv_id}")
            try:
                with open(cache_path, "rb") as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️ Failed to load cached PDF: {e}")
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
            print(f"⚠️ PDF download failed: {e}")
        
        return None
    
    def get_pdf_cache_path(self, paper_id: str, filename: str) -> str:
        """Get cache path for extracted image file (future-proof extra storage)."""
        safe_pid = paper_id.replace("/", "_").replace(".", "_")
        cache_subdir = os.path.join(PDF_CACHE_DIR, safe_pid)
        os.makedirs(cache_subdir, exist_ok=True)
        return os.path.join(cache_subdir, filename)
    
    def load_cached_pdf(self, paper_id: str, filename: str) -> Optional[bytes]:
        """Load extracted image from cache if available."""
        cache_path = self.get_pdf_cache_path(paper_id, filename)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️ Failed to load cached image {filename}: {e}")
        return None
    
    def save_pdf_to_cache(self, paper_id: str, filename: str, data: bytes) -> bool:
        """Save extracted image to cache for future runs."""
        cache_path = self.get_pdf_cache_path(paper_id, filename)
        try:
            with open(cache_path, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"⚠️ Failed to cache image {filename}: {e}")
            return False
    
    def filename_is_negative(self, img_path: str) -> bool:
        """Check if filename contains negative tokens."""
        fname = os.path.basename(img_path)
        tokens = self.preprocessor.preprocess_filename(fname)
        return any(t in FILENAME_NEGATIVE_TOKENS for t in tokens)
    
    def extract_figures_from_tex(self, tex: str) -> List[Figure]:
        r"""Extract figures from LaTeX text.
        Handles multiple \includegraphics within a single figure block and subfigures.
        """
        figures: List[Figure] = []

        for block in self.FIG_RE.findall(tex):
            cap = self.CAP_RE.search(block)
            if not cap:
                # No caption; skip to reduce false positives
                continue

            # Decode caption text safely
            try:
                from pylatexenc.latex2text import LatexNodes2Text
                caption_text = LatexNodes2Text().latex_to_text(cap.group(1))
            except Exception:
                caption_text = cap.group(1)

            caption_text = caption_text.strip()

            # Collect all image paths within the figure block
            img_paths = self.IMG_RE.findall(block) or []

            # If subfigures exist, also search within subfigure blocks explicitly
            for sub in self.SUBFIG_RE.findall(block):
                img_paths.extend(self.IMG_RE.findall(sub))

            # Attempt to extract subcaptions if present
            subcaptions = []
            # Collect explicit \subcaption{...}
            for sub in self.SUBFIG_RE.findall(block):
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

            # Create a Figure per image. If we have subcaptions and counts match, align them; else use shared caption.
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



        # Additionally: if any raw \Qcircuit blocks are embedded in the tex,
        # extract them live (single-file mode) and save under
        # `circuit_images/live_blocks/` using `core.live_latex_extractor`.
        # This is optional and non-blocking; failures are ignored.
        try:
            if _live_process_text and ('\\Qcircuit' in tex or self.QCIRCUIT_RE.search(tex)):
                _live_process_text(tex, source_name='inline_from_extract', render=True, render_with_module=True)
        except Exception:
            pass

        return figures



    def extract_images(self, tar: tarfile.TarFile, figures: List[Figure], 
                      paper_id: str) -> List[ExtractedImage]:
        """Extract images from tar file for selected figures."""
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
                        out_path = os.path.join(OUTPUT_DIR, fname)
                        
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
                        print(f"⚠️ Failed to extract {candidate}: {e}")
                    
                    break  # Stop trying extensions once found
        
        return extracted
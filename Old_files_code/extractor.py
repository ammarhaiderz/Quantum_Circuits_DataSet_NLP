"""
Figure extraction from LaTeX files and tar archives.
"""
import os
import tarfile
import io
from pylatexenc.latex2text import LatexNodes2Text

from config import OUTPUT_DIR, SUPPORTED_EXT
from preprocessor import TextPreprocessor


class FigureExtractor:
    """Handles figure extraction from arXiv sources."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def extract_figures_from_tex(self, tex):
        """
        Extract figures from LaTeX source code.
        """
        from preprocessor import FIG_RE, CAP_RE, IMG_RE
        
        figures = []
        for block in FIG_RE.findall(tex):
            cap = CAP_RE.search(block)
            img = IMG_RE.search(block)
            if not (cap and img):
                continue

            try:
                caption = LatexNodes2Text().latex_to_text(cap.group(1))
            except Exception:
                caption = cap.group(1)

            figures.append({
                "caption": caption.strip(),
                "img_path": img.group(1).strip()
            })
        return figures
    
    def extract_images(self, tar, figures, pid, saved_set):
        """
        Extract image files from tar archive.
        """
        members = {m.name: m for m in tar.getmembers()}
        extracted = []

        safe_pid = pid.replace("/", "_").replace(".", "_")

        for idx, f in enumerate(figures):
            base = f["img_path"]
            if self.preprocessor.filename_is_negative(base):
                continue
            for ext in [""] + SUPPORTED_EXT:
                candidate = base + ext
                if candidate in members:
                    key = f"{pid}:{candidate}"
                    if key in saved_set:
                        continue

                    data = tar.extractfile(members[candidate]).read()
                    fname = f"{safe_pid}_{idx}_{os.path.basename(candidate)}"
                    out = os.path.join(OUTPUT_DIR, fname)

                    with open(out, "wb") as w:
                        w.write(data)

                    saved_set.add(key)
                    extracted.append({
                        "file": out,
                        "img_name": os.path.basename(candidate),
                        "caption": f["caption"],
                        "preprocessed_text": f.get("preprocessed_text", ""),
                        "similarity": f.get("similarity", 0.0),
                        "sbert_sim": f.get("sbert_sim", 0.0),
                        "best_sbert_query": f.get("best_sbert_query", None)
                    })
                    break

        return extracted
"""
Data models for figures and extracted images.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os

@dataclass
class Figure:
    """Represents a figure extracted from a LaTeX document."""
    caption: str
    img_path: str
    paper_id: str = ""
    figure_number: int = 0
    panel: int = 0
    # Optional raw LaTeX drawing block (e.g., tikzpicture, circuitikz)
    latex_block: Optional[str] = None
    
    # TF-IDF fields
    preprocessed_text: str = ""
    similarities: Dict[str, float] = field(default_factory=dict)
    best_query: Optional[str] = None
    similarity_raw: float = 0.0
    similarity: float = 0.0
    negative_tokens: int = 0
    penalty: float = 0.0
    
    # SBERT fields
    sbert_sim: float = 0.0
    best_sbert_query: Optional[str] = None
    
    # Combined scoring
    combined_score: float = 0.0
    
    # Selection flags
    selected: bool = False
    extracted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for tabular export.

        Returns
        -------
        dict
            Dictionary with caption, similarity, SBERT, selection, and metadata fields.
        """
        return {
            "paper_id": self.paper_id,
            "figure_number": self.figure_number,
            "panel": self.panel,
            "img_name": os.path.basename(self.img_path),
            "raw_caption": self.caption,
            "preprocessed_text": self.preprocessed_text,
            "similarity": self.similarity,
            "similarity_raw": self.similarity_raw,
            "negative_tokens": self.negative_tokens,
            "penalty": self.penalty,
            "best_query": self.best_query,
            "sbert_sim": self.sbert_sim,
            "best_sbert_query": self.best_sbert_query,
            "selected": self.selected,
            "extracted": self.extracted,
            "has_latex": self.latex_block is not None,
            **{f"sim_{k}": v for k, v in self.similarities.items()}
        }

@dataclass
class ExtractedImage:
    """Represents an extracted image file."""
    file_path: str
    img_name: str
    caption: str
    preprocessed_text: str
    similarity: float
    sbert_sim: float
    best_sbert_query: Optional[str]
    paper_id: str = ""
    
    @property
    def filename(self):
        """Return the basename of the image file path."""
        return os.path.basename(self.file_path)
"""
Figure data models for quantum circuit extraction.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os

@dataclass
class ExtractedImage:
    """Represents an extracted image with metadata (for compatibility)."""
    caption: str = ""
    image_path: str = ""
    paper_id: str = ""
    figure_number: int = 0
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'caption': self.caption,
            'image_path': self.image_path,
            'paper_id': self.paper_id,
            'figure_number': self.figure_number,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractedImage':
        """Create from dictionary."""
        return cls(
            caption=data.get('caption', ''),
            image_path=data.get('image_path', ''),
            paper_id=data.get('paper_id', ''),
            figure_number=data.get('figure_number', 0),
            confidence=data.get('confidence', 0.0),
            metadata=data.get('metadata', {})
        )

class Figure:
    """Represents a figure with its caption and metadata for NLP processing."""
    
    def __init__(self, caption: str = "", image_path: str = "", paper_id: str = ""):
        self.caption = caption
        self.image_path = image_path
        self.paper_id = paper_id
        
        # TF-IDF scores
        self.preprocessed_text = ""
        self.similarity = 0.0
        self.similarity_raw = 0.0
        self.best_query = None
        self.negative_penalty = 0.0
        
        # SBERT scores
        self.sbert_sim = 0.0
        self.best_sbert_query = None
        self.cross_scores = {}
        
        # Zero-shot scores
        self.zero_shot_is_circuit = False
        self.zero_shot_confidence = 0.0
        self.zero_shot_label = ""
        
        # Combined score
        self.combined_score = 0.0
        
        # Original extraction metadata
        self.extraction_metadata: Dict[str, Any] = {}
        
    def __repr__(self):
        caption_preview = self.caption[:50] + "..." if len(self.caption) > 50 else self.caption
        return f"Figure(paper={self.paper_id}, caption='{caption_preview}')"
    
    def to_extracted_image(self) -> ExtractedImage:
        """Convert to ExtractedImage for compatibility with existing pipeline."""
        return ExtractedImage(
            caption=self.caption,
            image_path=self.image_path,
            paper_id=self.paper_id,
            metadata={
                'similarity': self.similarity,
                'sbert_sim': self.sbert_sim,
                'best_query': self.best_query,
                'best_sbert_query': self.best_sbert_query,
                'combined_score': self.combined_score,
                'zero_shot_confidence': self.zero_shot_confidence,
                'negative_penalty': self.negative_penalty,
                'preprocessed_text': self.preprocessed_text
            }
        )
    
    @classmethod
    def from_extracted_image(cls, extracted: ExtractedImage) -> 'Figure':
        """Create Figure from ExtractedImage."""
        figure = cls(
            caption=extracted.caption,
            image_path=extracted.image_path,
            paper_id=extracted.paper_id
        )
        
        # Copy metadata if available
        if extracted.metadata:
            figure.extraction_metadata = extracted.metadata.copy()
            figure.similarity = extracted.metadata.get('similarity', 0.0)
            figure.sbert_sim = extracted.metadata.get('sbert_sim', 0.0)
            figure.best_query = extracted.metadata.get('best_query')
            figure.best_sbert_query = extracted.metadata.get('best_sbert_query')
            figure.combined_score = extracted.metadata.get('combined_score', 0.0)
            figure.zero_shot_confidence = extracted.metadata.get('zero_shot_confidence', 0.0)
            figure.negative_penalty = extracted.metadata.get('negative_penalty', 0.0)
            figure.preprocessed_text = extracted.metadata.get('preprocessed_text', "")
        
        return figure
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Figure':
        """Create Figure from dictionary."""
        figure = cls(
            caption=data.get('caption', ''),
            image_path=data.get('image_path', ''),
            paper_id=data.get('paper_id', '')
        )
        
        # Set scores if available
        figure.similarity = data.get('similarity', 0.0)
        figure.sbert_sim = data.get('sbert_sim', 0.0)
        figure.best_query = data.get('best_query')
        figure.best_sbert_query = data.get('best_sbert_query')
        figure.combined_score = data.get('combined_score', 0.0)
        figure.zero_shot_confidence = data.get('zero_shot_confidence', 0.0)
        figure.zero_shot_label = data.get('zero_shot_label', '')
        figure.negative_penalty = data.get('negative_penalty', 0.0)
        figure.preprocessed_text = data.get('preprocessed_text', '')
        
        return figure
    
    def to_dict(self) -> Dict:
        """Convert Figure to dictionary."""
        return {
            'caption': self.caption,
            'image_path': self.image_path,
            'paper_id': self.paper_id,
            'similarity': self.similarity,
            'sbert_sim': self.sbert_sim,
            'best_query': self.best_query,
            'best_sbert_query': self.best_sbert_query,
            'combined_score': self.combined_score,
            'zero_shot_confidence': self.zero_shot_confidence,
            'zero_shot_label': self.zero_shot_label,
            'negative_penalty': self.negative_penalty,
            'preprocessed_text': self.preprocessed_text
        }
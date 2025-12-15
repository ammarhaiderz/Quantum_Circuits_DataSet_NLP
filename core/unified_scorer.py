"""
Unified scoring logic for consistent ranking.
"""

from typing import List
import numpy as np

from models.figure_data import Figure
from config.settings import (
    USE_COMBINED_SCORE,
    COMBINED_SCORING_STRATEGY,
    TFIDF_WEIGHT,
    SBERT_WEIGHT,
    COMBINED_THRESHOLD,
    SBERT_MIN_SIM,
    SIMILARITY_THRESHOLD
)

class UnifiedScorer:
    """Consistent scoring approach."""
    
    def __init__(self, strategy=COMBINED_SCORING_STRATEGY):
        """
        Initialize scorer with chosen strategy.
        
        Strategies:
        - "combined": Weighted combination of TF-IDF and SBERT scores
        - "cascade": TF-IDF threshold → SBERT threshold → SBERT ranking
        - "sbert_only": SBERT score only
        """
        self.strategy = strategy
        
        if strategy == "combined":
            print(f"🎯 Using combined scoring (TF-IDF: {TFIDF_WEIGHT}, SBERT: {SBERT_WEIGHT})")
        elif strategy == "cascade":
            print(f"🎯 Using cascade scoring (TF-IDF ≥ {SIMILARITY_THRESHOLD}, SBERT ≥ {SBERT_MIN_SIM})")
        else:
            print(f"🎯 Using SBERT-only scoring")
    
    def score_figures(self, figures: List[Figure]) -> List[Figure]:
        """Apply consistent scoring strategy."""
        if not figures:
            return []
        
        if self.strategy == "combined":
            return self._combined_scoring(figures)
        elif self.strategy == "cascade":
            return self._cascade_scoring(figures)
        else:  # sbert_only
            return self._sbert_only_scoring(figures)
    
    def _combined_scoring(self, figures: List[Figure]) -> List[Figure]:
        """Weighted combination of TF-IDF and SBERT scores."""
        for fig in figures:
            # Normalize scores to similar ranges
            tfidf_norm = self._normalize_tfidf(fig.similarity)
            sbert_norm = self._normalize_sbert(fig.sbert_sim)
            
            # Calculate combined score
            fig.combined_score = (
                TFIDF_WEIGHT * tfidf_norm + 
                SBERT_WEIGHT * sbert_norm
            )
        
        # Sort by combined score
        figures.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Apply threshold
        filtered = [f for f in figures if f.combined_score >= COMBINED_THRESHOLD]
        
        print(f"📊 Combined scoring: {len(filtered)}/{len(figures)} above threshold {COMBINED_THRESHOLD}")
        return filtered
    
    def _cascade_scoring(self, figures: List[Figure]) -> List[Figure]:
        """Two-stage cascade filtering."""
        # Stage 1: TF-IDF threshold
        tfidf_accepted = [f for f in figures if f.similarity >= SIMILARITY_THRESHOLD]
        print(f"📊 Cascade stage 1 (TF-IDF): {len(tfidf_accepted)}/{len(figures)} passed")
        
        # Stage 2: SBERT threshold
        sbert_accepted = [f for f in tfidf_accepted if f.sbert_sim >= SBERT_MIN_SIM]
        print(f"📊 Cascade stage 2 (SBERT): {len(sbert_accepted)}/{len(tfidf_accepted)} passed")
        
        # Sort by SBERT score
        sbert_accepted.sort(key=lambda x: x.sbert_sim, reverse=True)
        
        return sbert_accepted
    
    def _sbert_only_scoring(self, figures: List[Figure]) -> List[Figure]:
        """SBERT score only."""
        # Sort by SBERT score
        figures.sort(key=lambda x: x.sbert_sim, reverse=True)
        
        # Optional: Apply threshold
        if SBERT_MIN_SIM > 0:
            filtered = [f for f in figures if f.sbert_sim >= SBERT_MIN_SIM]
            print(f"📊 SBERT-only: {len(filtered)}/{len(figures)} above threshold {SBERT_MIN_SIM}")
            return filtered
        
        return figures
    
    def _normalize_tfidf(self, score: float) -> float:
        """Normalize TF-IDF score to [0, 1] range."""
        # TF-IDF scores are already roughly in [0, 1]
        # Apply mild scaling
        return min(1.0, score * 1.5)
    
    def _normalize_sbert(self, score: float) -> float:
        """Normalize SBERT score to [0, 1] range."""
        # SBERT cosine similarity is in [-1, 1], but typically [0.2, 0.95]
        # Map to [0, 1]
        return max(0.0, min(1.0, (score - 0.2) / 0.7))
    
    def get_top_figures(self, figures: List[Figure], top_k: int) -> List[Figure]:
        """Get top N figures based on scoring strategy."""
        scored = self.score_figures(figures)
        return scored[:top_k]
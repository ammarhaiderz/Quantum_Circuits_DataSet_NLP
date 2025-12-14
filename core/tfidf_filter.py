"""
TF-IDF filtering logic for figure selection.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

from models.figure_data import Figure
from core.preprocessor import TextPreprocessor
from config.queries import QUERY_SETS
from config.settings import USE_NEGATIVE_PENALTY, NEGATIVE_PENALTY_ALPHA, SIMILARITY_THRESHOLD


class TfidfFilter:
    """Handles TF-IDF filtering of figures."""
    
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self._build_allowed_vocab()
    
    def _build_allowed_vocab(self):
        """Build allowed vocabulary from all query sets."""
        all_query_text = "\n".join(QUERY_SETS.values())
        self.ALLOWED_VOCAB = set(self.preprocessor.tfidf_analyzer(all_query_text))
    
    def filter_figures(self, figures: List[Figure]) -> List[Figure]:
        """Apply TF-IDF filtering to figures."""
        texts = [f.caption for f in figures]
        
        vectorizer = TfidfVectorizer(
            analyzer=self.preprocessor.tfidf_analyzer,
            vocabulary=self.ALLOWED_VOCAB,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            norm='l2'
        )
        
        tfidf = vectorizer.fit_transform(texts + list(QUERY_SETS.values()))
        caption_vecs = tfidf[:len(texts)]
        query_vecs = tfidf[len(texts):]
        
        sims = cosine_similarity(caption_vecs, query_vecs)
        
        for i, figure in enumerate(figures):
            preproc = self.preprocessor.preprocess_text_to_string(figure.caption)
            
            per_query = {
                name: float(sims[i, j])
                for j, name in enumerate(QUERY_SETS.keys())
            }
            best_sim = max(per_query.values())
            best_query = max(per_query, key=per_query.get)
            
            # Negative token penalty
            neg_count = self.preprocessor.count_negative_tokens(preproc)
            
            if USE_NEGATIVE_PENALTY:
                penalty = NEGATIVE_PENALTY_ALPHA * neg_count
            else:
                penalty = 0.0
            
            adjusted_sim = max(0.0, best_sim - penalty)
            
            figure.preprocessed_text = preproc
            figure.similarities = per_query
            figure.best_query = best_query
            figure.similarity_raw = best_sim
            figure.negative_tokens = neg_count
            figure.penalty = penalty
            figure.similarity = adjusted_sim
        
        return figures
    
    def get_accepted_figures(self, figures: List[Figure]) -> List[Figure]:
        # """Get figures that pass the TF-IDF threshold."""
        # return [f for f in figures if f.similarity >= SIMILARITY_THRESHOLD]
        """Get figures that pass the TF-IDF threshold."""
        # Filter by threshold AND minimum token overlap
        MIN_TOKEN_OVERLAP = 2  # Require at least 2 matching tokens
        accepted = []
        for f in figures:
            if f.similarity >= SIMILARITY_THRESHOLD:
                # Count tokens that match allowed vocab
                tokens = set(f.preprocessed_text.split())
                overlap = len(tokens & self.ALLOWED_VOCAB)
                if overlap >= MIN_TOKEN_OVERLAP:
                    accepted.append(f)
        return accepted
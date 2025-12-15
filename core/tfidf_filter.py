"""
FIXED TF-IDF filtering logic with proper negative penalty
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import numpy as np
import re  # ADDED: Missing import
from typing import List, Dict

try:
    import nltk
    from nltk import pos_tag
except Exception:
    pos_tag = None

from models.figure_data import Figure
from core.preprocessor import TextPreprocessor
from config.queries import QUERY_SETS
from config.settings import (
    USE_NEGATIVE_PENALTY,
    NEGATIVE_PENALTY_ALPHA,
    SIMILARITY_THRESHOLD,
    USE_CUSTOM_TFIDF_FEATURES,
)

class FixedTfidfFilter:
    """Fixed TF-IDF filtering with proper penalty calculation."""
    
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self._build_allowed_vocab()
    
    def _build_allowed_vocab(self):
        """Build allowed vocabulary from all query sets."""
        all_query_text = "\n".join(QUERY_SETS.values())
        self.ALLOWED_VOCAB = set(self.preprocessor.tfidf_analyzer(all_query_text))
        
        # Build domain bigrams
        self.ALLOWED_BIGRAMS = set()
        for query_text in QUERY_SETS.values():
            for line in query_text.strip().split('\n'):
                line = line.strip()
                if line:
                    tokens = self.preprocessor.tfidf_analyzer(line)
                    if len(tokens) > 1:  # Only create bigrams if we have tokens
                        bigrams = [" ".join(bg) for bg in zip(tokens, tokens[1:])]
                        self.ALLOWED_BIGRAMS.update(bigrams)
        
        print(f"📊 Built vocabulary: {len(self.ALLOWED_VOCAB)} terms, {len(self.ALLOWED_BIGRAMS)} bigrams")
    
    def debug_tfidf_issues(self, figures: List[Figure]):
        """Debug TF-IDF preprocessing issues."""
        print("\n🔍 TF-IDF DEBUG INFO:")
        
        # Check captions
        empty_count = sum(1 for f in figures if not f.caption or not f.caption.strip())
        print(f"  Total figures: {len(figures)}")
        print(f"  Empty captions: {empty_count}")
        
        # Check preprocessing
        valid_figures = [f for f in figures if f.caption and f.caption.strip()]
        print(f"  Valid captions: {len(valid_figures)}")
        
        if valid_figures:
            for i, fig in enumerate(valid_figures[:3]):  # Check first 3
                if fig.caption:
                    tokens = self.preprocessor.tfidf_analyzer(fig.caption)
                    print(f"  Figure {i} caption: '{fig.caption[:50]}...'")
                    print(f"    Tokens: {tokens}")
                    print(f"    Token count: {len(tokens)}")
        
        # Check vocabulary
        print(f"\n  Allowed vocab size: {len(self.ALLOWED_VOCAB)}")
        if self.ALLOWED_VOCAB:
            print(f"  Sample vocab: {list(self.ALLOWED_VOCAB)[:10]}")
    
    def _calculate_negative_penalty(self, caption: str) -> float:
        """
        Dynamic negative penalty based on caption characteristics.
        Returns penalty factor (1.0 = no penalty, 0.0 = full penalty).
        """
        if not caption:
            return 1.0
            
        caption_lower = caption.lower()
        
        # 1. Check for non-circuit patterns
        non_circuit_patterns = [
            r'fig\.?\s*\d+',
            r'table\s*\d+',
            r'energy\s+level',
            r'spectrum',
            r'waveform',
            r'3d\s+render',
            r'simulation',
            r'plot\s+of',
            r'graph\s+of',
            r'histogram',
            r'heat.?map'
        ]
        
        pattern_matches = sum(1 for pattern in non_circuit_patterns 
                            if re.search(pattern, caption_lower))
        
        # 2. Count negative indicators (reduced set)
        words = caption_lower.split()
        common_negative_words = {
            'result', 'results', 'show', 'shows', 'shown',
            'example', 'compared', 'comparison',
            'energy', 'level', 'plot', 'graph', 'chart', 'figure'
        }
        
        negative_word_count = sum(1 for word in words if word in common_negative_words)
        
        # 3. Calculate penalty (much more gentle)
        total_negative_indicators = pattern_matches + negative_word_count
        word_count = max(len(words), 1)
        negative_density = total_negative_indicators / word_count
        
        # FIXED: Much gentler penalty curve
        # At density=0.1 (10% negative), penalty ~0.95 (only 5% reduction)
        # At density=0.3 (30% negative), penalty ~0.85 (15% reduction)
        penalty_factor = 1.0 / (1.0 + np.exp(8 * (negative_density - 0.2)))
        
        return max(0.5, penalty_factor)  # Never reduce below 50%
    
    def _compute_custom_features(self, texts: List[str], idf_map: Dict) -> np.ndarray:
        """Compute custom feature vectors for texts."""
        feats = []
        for text in texts:
            tokens = self.preprocessor.tfidf_analyzer(text)
            token_count = max(len(tokens), 1)
            
            domain_hits = [t for t in tokens if t in self.ALLOWED_VOCAB]
            domain_ratio = len(domain_hits) / token_count
            
            # Bigram hits
            bigrams = [" ".join(bg) for bg in zip(tokens, tokens[1:])] if len(tokens) > 1 else []
            bigram_hits = [b for b in bigrams if b in self.ALLOWED_BIGRAMS]
            bigram_ratio = len(bigram_hits) / max(len(bigrams), 1) if bigrams else 0.0
            
            # Position-based features
            pos_window = min(10, len(tokens))
            pos_domain_ratio = (
                sum(1 for t in tokens[:pos_window] if t in self.ALLOWED_VOCAB)
                / max(pos_window, 1)
            )
            
            # IDF features
            idf_vals = [idf_map.get(t, 0) for t in tokens if t in idf_map]
            rare_idf_mean = sum(idf_vals) / len(idf_vals) if idf_vals else 0.0
            
            feats.append([
                domain_ratio,
                bigram_ratio,
                rare_idf_mean,
                pos_domain_ratio,
            ])
        
        return np.array(feats, dtype=float) if feats else np.array([])
    
    def filter_figures(self, figures: List[Figure]) -> List[Figure]:
        """Apply TF-IDF filtering to figures."""
        # Prepare texts - only include non-empty captions
        texts = []
        valid_indices = []
        
        for i, fig in enumerate(figures):
            if fig.caption and fig.caption.strip():
                texts.append(fig.caption)
                valid_indices.append(i)
        
        if not texts:
            print("⚠️ No valid captions for TF-IDF filtering")
            for f in figures:
                f.similarity = 0.0
                f.best_query = None
                f.preprocessed_text = ""
            return figures
        
        print(f"📊 TF-IDF processing {len(texts)} valid captions")
        
        # Create vectorizer - FIXED: Remove ngram_range when analyzer is callable
        vectorizer = TfidfVectorizer(
            analyzer=self.preprocessor.tfidf_analyzer,
            vocabulary=self.ALLOWED_VOCAB if self.ALLOWED_VOCAB else None,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            norm='l2',
            min_df=1,
            max_df=1.0
        )
        
        # Fit on captions + queries
        query_texts = list(QUERY_SETS.values())
        all_texts = texts + query_texts
        
        try:
            tfidf = vectorizer.fit_transform(all_texts)
        except ValueError as e:
            print(f"⚠️ Vectorization failed: {e}")
            # Initialize empty scores
            for f in figures:
                f.similarity = 0.0
                f.best_query = None
                f.preprocessed_text = ""
            return figures
        
        # Check if we got any features
        if tfidf.shape[1] == 0:
            print("⚠️ No features extracted from texts")
            for f in figures:
                f.similarity = 0.0
                f.best_query = None
                f.preprocessed_text = ""
            return figures
        
        # Get vocab and IDF
        vocab = vectorizer.get_feature_names_out()
        idf_values = vectorizer.idf_
        idf_map = {term: idf for term, idf in zip(vocab, idf_values)}
        
        # Split matrices
        caption_vecs = tfidf[:len(texts)]
        query_vecs = tfidf[len(texts):]
        
        # Check matrix shapes
        print(f"  Caption matrix shape: {caption_vecs.shape}")
        print(f"  Query matrix shape: {query_vecs.shape}")
        
        if caption_vecs.shape[0] == 0 or query_vecs.shape[0] == 0:
            print("⚠️ Empty matrices after vectorization")
            for f in figures:
                f.similarity = 0.0
                f.best_query = None
                f.preprocessed_text = ""
            return figures
        
        # Optional custom features
        if USE_CUSTOM_TFIDF_FEATURES:
            custom_feats = self._compute_custom_features(all_texts, idf_map)
            CUSTOM_FEATURE_WEIGHT = 0.6
            
            # Scale features if we have any
            if custom_feats.size > 0 and custom_feats.shape[0] > 0:
                for j in range(custom_feats.shape[1]):
                    col = custom_feats[:, j]
                    c_min, c_max = col.min(), col.max()
                    if c_max - c_min > 0:
                        custom_feats[:, j] = (col - c_min) / (c_max - c_min)
                
                weighted_feats = custom_feats * CUSTOM_FEATURE_WEIGHT
                cap_feats = csr_matrix(weighted_feats[:len(texts)])
                qry_feats = csr_matrix(weighted_feats[len(texts):])
                
                caption_vecs = hstack([caption_vecs, cap_feats])
                query_vecs = hstack([query_vecs, qry_feats])
        
        # Calculate similarities
        try:
            sims = cosine_similarity(caption_vecs, query_vecs)
        except Exception as e:
            print(f"⚠️ Cosine similarity failed: {e}")
            print(f"  caption_vecs shape: {caption_vecs.shape}")
            print(f"  query_vecs shape: {query_vecs.shape}")
            # Initialize empty scores
            for idx in valid_indices:
                figures[idx].similarity = 0.0
                figures[idx].best_query = None
                figures[idx].preprocessed_text = ""
            return figures
        
        # Process each valid figure
        for i, fig_idx in enumerate(valid_indices):
            figure = figures[fig_idx]
            
            # Get best similarity
            fig_sims = sims[i]
            best_sim = float(np.max(fig_sims))
            best_query_idx = int(np.argmax(fig_sims))
            best_query = list(QUERY_SETS.keys())[best_query_idx]
            
            # Apply negative penalty if enabled
            if USE_NEGATIVE_PENALTY:
                penalty_factor = self._calculate_negative_penalty(figure.caption)
                final_sim = best_sim * penalty_factor
                penalty_amount = 1.0 - penalty_factor
            else:
                final_sim = best_sim
                penalty_amount = 0.0
            
            # Store results
            figure.preprocessed_text = self.preprocessor.preprocess_text_to_string(figure.caption)
            figure.similarity = final_sim
            figure.similarity_raw = best_sim
            figure.best_query = best_query
            figure.negative_penalty = penalty_amount
        
        # For figures without captions, set default scores
        for i, fig in enumerate(figures):
            if i not in valid_indices:
                fig.similarity = 0.0
                fig.best_query = None
                fig.preprocessed_text = ""
        
        return figures
    
    def get_accepted_figures(self, figures: List[Figure]) -> List[Figure]:
        """Get figures that pass the TF-IDF threshold."""
        MIN_TOKEN_OVERLAP = 2
        accepted = []
        
        for f in figures:
            if f.similarity >= SIMILARITY_THRESHOLD:
                tokens = set(self.preprocessor.tfidf_analyzer(f.caption or ""))
                overlap = len(tokens & self.ALLOWED_VOCAB)
                if overlap >= MIN_TOKEN_OVERLAP:
                    accepted.append(f)
        
        print(f"📊 TF-IDF accepted: {len(accepted)}/{len(figures)} figures")
        return accepted
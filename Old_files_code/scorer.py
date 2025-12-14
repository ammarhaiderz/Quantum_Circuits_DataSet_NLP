"""
TF-IDF scoring module for initial filtering.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    QUERY_SETS
)
from preprocessor import TextPreprocessor


class TFIDFScorer:
    """Handles TF-IDF scoring and filtering."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        
    def build_vocabulary(self):
        """Build vocabulary from query sets."""
        all_query_text = "\n".join(QUERY_SETS.values())
        return set(self.preprocessor.tfidf_analyzer(all_query_text))
    
    def fit_vectorizer(self, texts):
        """Fit TF-IDF vectorizer on texts and queries."""
        ALLOWED_VOCAB = self.build_vocabulary()
        
        self.vectorizer = TfidfVectorizer(
            analyzer=self.preprocessor.tfidf_analyzer,
            vocabulary=ALLOWED_VOCAB,
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            norm='l2'
        )
        
        # Fit on captions + queries
        all_texts = texts + list(QUERY_SETS.values())
        return self.vectorizer.fit_transform(all_texts)
    
    def score_figures(self, figures):
        """Score figures using TF-IDF."""
        if not figures:
            return figures
        
        texts = [f["caption"] for f in figures]
        
        # Fit vectorizer and transform
        tfidf_matrix = self.fit_vectorizer(texts)
        caption_vecs = tfidf_matrix[:len(texts)]
        query_vecs = tfidf_matrix[len(texts):]
        
        # Compute similarities
        sims = cosine_similarity(caption_vecs, query_vecs)
        
        # Score each figure
        for i, f in enumerate(figures):
            preproc = self.preprocessor.preprocess_text_to_string(f["caption"])

            # Check for hard rejection
            is_hard_rejected = self.preprocessor.check_hard_rejection(
                f["caption"]
            )

            # Per-query similarities
            per_query = {
                name: float(sims[i, j])
                for j, name in enumerate(QUERY_SETS.keys())
            }
            best_sim = max(per_query.values())
            best_query = max(per_query, key=per_query.get)

            # Apply context-aware negative penalty
            penalty = (
                self.preprocessor.calculate_context_aware_penalty(
                    f["caption"]
                )
            )

            # Hard rejection: set score to 0
            if is_hard_rejected:
                adjusted_sim = 0.0
            else:
                # Soft penalty: reduce by context-aware amount
                adjusted_sim = max(0.0, best_sim - penalty)

            # Update figure with scores
            f.update({
                "preprocessed_text": preproc,
                "similarities": per_query,
                "best_query": best_query,
                "similarity_raw": best_sim,
                "negative_tokens": self.preprocessor.count_negative_tokens(
                    preproc
                ),
                "penalty": penalty,
                "hard_rejected": is_hard_rejected,
                "similarity": adjusted_sim
            })
        
        return figures


class EnhancedTFIDFScorer:
    def __init__(self):
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Capture 1-3 grams
            max_features=5000
        )
        
        # Negative word weights
        self.negative_weights = {
            "classical": -3.0,
            "electrical": -2.5,
            "digital": -2.5,
            "analog": -2.5,
            "transistor": -2.0,
            "logic": -2.0,
            "pcb": -2.0,
            "cmos": -2.0,
            "resistor": -2.0,
            "capacitor": -2.0,
            "inductor": -2.0,
            "voltage": -1.5,
            "current": -1.5,
            "power": -1.5,
            "board": -1.5,
            "wire": -1.5,
            "connection": -1.5,
            "boolean": -1.5,
        }
        
        # Negative phrase weights
        self.negative_phrases = {
            "classical circuit": -5.0,
            "electrical circuit": -5.0,
            "digital circuit": -5.0,
            "logic gate": -4.0,
            "circuit board": -4.0,
            "power supply": -3.0,
            "schematic symbol": -3.0,
            "wire connection": -3.0,
        }
        
        self.is_fitted = False
        self.query_vectors = None
        self.query_texts = []
    
    def create_query_vectors(self):
        """Create query vectors from QUERY_SETS."""
        query_texts = []
        
        # Extract all query lines from QUERY_SETS
        for category, query_block in QUERY_SETS.items():
            lines = [
                line.strip()
                for line in query_block.strip().split('\n')
                if line.strip()
            ]
            query_texts.extend(lines)
        
        # Remove duplicates
        self.query_texts = list(set(query_texts))
        
        print(
            "DEBUG: Creating query vectors from "
            f"{len(self.query_texts)} unique queries"
        )
        
        # Fit vectorizer on query texts
        self.vectorizer.fit(self.query_texts)
        self.query_vectors = self.vectorizer.transform(self.query_texts)
        self.is_fitted = True
    
    def calculate_penalty(self, text):
        """Calculate penalty based on negative indicators."""
        text_lower = text.lower()
        penalty = 0.0
        
        # Word-level penalties
        words = text_lower.split()
        for word in words:
            if word in self.negative_weights:
                penalty += self.negative_weights[word]
        
        # Phrase-level penalties
        for phrase, weight in self.negative_phrases.items():
            if phrase in text_lower:
                penalty += weight
        
        return penalty
    
    def score_figures(self, figures):
        """Score figures with enhanced TF-IDF and penalties."""
        if not figures:
            return figures
        
        if not self.is_fitted:
            self.create_query_vectors()
        
        # Extract and preprocess captions
        captions = []
        valid_indices = []
        
        for i, fig in enumerate(figures):
            # Make sure figure has preprocessed_text
            if "preprocessed_text" not in fig:
                # If not, create it from caption
                if "caption" in fig:
                    fig["preprocessed_text"] = fig["caption"].lower()
                    captions.append(fig["preprocessed_text"])
                    valid_indices.append(i)
                else:
                    # No caption, skip this figure
                    continue
            elif fig["preprocessed_text"]:
                captions.append(fig["preprocessed_text"])
                valid_indices.append(i)
        
        if not captions:
            print("DEBUG: No valid captions to score")
            return figures
        
        print(f"DEBUG: Scoring {len(captions)} figures with captions")
        
        # Transform captions
        caption_vectors = self.vectorizer.transform(captions)
        
        # Calculate similarities
        similarities = cosine_similarity(caption_vectors, self.query_vectors)
        
        # Get max similarity for each caption
        max_similarities = similarities.max(axis=1)
        best_query_indices = similarities.argmax(axis=1)
        
        # Apply scores back to figures
        for idx, fig_idx in enumerate(valid_indices):
            figure = figures[fig_idx]
            
            # Calculate penalty
            penalty = self.calculate_penalty(figure["preprocessed_text"])
            
            # Apply penalty (reduce similarity)
            raw_similarity = max_similarities[idx]
            penalized_similarity = max(0, raw_similarity - (penalty * 0.1))
            
            # Get best matching query
            best_query_idx = best_query_indices[idx]
            best_query = (
                self.query_texts[best_query_idx]
                if best_query_idx < len(self.query_texts)
                else ""
            )
            
            # Store results - MAKE SURE THESE KEYS EXIST
            figure["similarity"] = penalized_similarity
            figure["similarity_raw"] = raw_similarity
            figure["penalty"] = penalty
            figure["best_query"] = best_query
            
            # Store per-query similarities
            figure["similarities"] = {}
            for j, query_text in enumerate(self.query_texts):
                key = f"query_{j}"
                figure["similarities"][key] = float(similarities[idx, j])
            
            # Track negative tokens found
            figure["negative_tokens"] = []
            words = figure["preprocessed_text"].lower().split()
            for word in words:
                if word in self.negative_weights:
                    figure["negative_tokens"].append(word)
            
            print(
                "DEBUG: Figure "
                f"{idx} - similarity: {penalized_similarity:.4f}, "
                f"penalty: {penalty:.2f}"
            )
        
        return figures

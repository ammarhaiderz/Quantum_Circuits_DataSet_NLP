"""
TF-IDF filtering logic for figure selection.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import numpy as np
from typing import List

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


class TfidfFilter:
    """Handles TF-IDF filtering of figures."""
    
    # Lightweight dictionary-based NER for domain entities
    GATE_TERMS = {
        "cnot", "cx", "cz", "toffoli", "swap", "hadamard", "pauli", "x", "y", "z"
    }
    GATE_PHRASES = {
        "x gate", "z gate", "y gate", "cnot gate", "controlled not", "controlled z",
        "controlled x", "cz gate", "cx gate", "swap gate", "toffoli gate"
    }
    CIRCUIT_TERMS = {
        "qubit", "qubits", "register", "wire", "wires", "ancilla", "ancillae",
        "circuit", "circuitry", "diagram"
    }
    CIRCUIT_PHRASES = {
        "quantum circuit", "circuit diagram", "gate sequence", "circuit depth",
        "quantum register", "qubit register", "wiring diagram"
    }
    ALGO_TERMS = {
        "qft", "grover", "shor", "simon", "oracle", "ansatz", "vqe", "qaoa",
        "variational", "fourier"
    }
    ALGO_PHRASES = {
        "quantum fourier transform", "deutsch jozsa", "bernstein vazirani",
        "grover search", "quantum counting", "variational quantum eigensolver",
        "quantum approximate optimization"
    }

    def __init__(self, preprocessor: TextPreprocessor):
        """Initialize filter with a text preprocessor.

        Parameters
        ----------
        preprocessor : TextPreprocessor
            Preprocessor used for tokenization and normalization.
        """
        self.preprocessor = preprocessor
        self._build_allowed_vocab()
    
    def _build_allowed_vocab(self):
        """Build allowed vocabulary and bigrams from all query sets."""
        all_query_text = "\n".join(QUERY_SETS.values())
        self.ALLOWED_VOCAB = set(self.preprocessor.tfidf_analyzer(all_query_text))
        # Build domain bigrams for specificity scoring (per-line to avoid boundary bigrams)
        self.ALLOWED_BIGRAMS = set()
        for query_text in QUERY_SETS.values():
            for line in query_text.strip().split('\n'):
                line = line.strip()
                if line:
                    tokens = self.preprocessor.tfidf_analyzer(line)
                    bigrams = [" ".join(bg) for bg in zip(tokens, tokens[1:])]
                    self.ALLOWED_BIGRAMS.update(bigrams)

    def _compute_custom_features(self, texts: List[str], idf_map) -> np.ndarray:
        """Compute custom feature vectors for texts.

        Parameters
        ----------
        texts : list[str]
            Input texts to featurize.
        idf_map : dict
            Mapping from token to inverse document frequency.

        Returns
        -------
        numpy.ndarray
            Feature array of shape ``(n_texts, n_features)``.
        """
        feats = []
        for text in texts:
            tokens = self.preprocessor.tfidf_analyzer(text)
            token_count = max(len(tokens), 1)
            preproc = self.preprocessor.preprocess_text_to_string(text)
            neg_count = self.preprocessor.count_negative_tokens(preproc)

            domain_hits = [t for t in tokens if t in self.ALLOWED_VOCAB]
            domain_ratio = len(domain_hits) / token_count

            bigrams = [" ".join(bg) for bg in zip(tokens, tokens[1:])]
            bigram_hits = [b for b in bigrams if b in self.ALLOWED_BIGRAMS]
            bigram_ratio = len(bigram_hits) / max(len(bigrams), 1) if bigrams else 0.0

            pos_window = min(10, len(tokens))
            pos_domain_ratio = (
                sum(1 for t in tokens[:pos_window] if t in self.ALLOWED_VOCAB)
                / max(pos_window, 1)
            )

            idf_vals = [idf_map[t] for t in tokens if t in idf_map]
            rare_idf_mean = sum(idf_vals) / len(idf_vals) if idf_vals else 0.0

            neg_ratio = neg_count / token_count
            is_short = 1.0 if len(tokens) < 6 else 0.0

            # Dictionary-based NER-style signals
            lower_text = text.lower()
            gate_hits = [t for t in tokens if t in self.GATE_TERMS]
            gate_phrase_hits = [p for p in self.GATE_PHRASES if p in lower_text]
            circuit_hits = [t for t in tokens if t in self.CIRCUIT_TERMS]
            circuit_phrase_hits = [p for p in self.CIRCUIT_PHRASES if p in lower_text]
            algo_hits = [t for t in tokens if t in self.ALGO_TERMS]
            algo_phrase_hits = [p for p in self.ALGO_PHRASES if p in lower_text]

            ner_gate_count = len(gate_hits) + len(gate_phrase_hits)
            ner_entity_count = (
                ner_gate_count
                + len(circuit_hits)
                + len(circuit_phrase_hits)
                + len(algo_hits)
                + len(algo_phrase_hits)
            )
            ner_unique_entities = len(
                set(gate_hits)
                | set(gate_phrase_hits)
                | set(circuit_hits)
                | set(circuit_phrase_hits)
                | set(algo_hits)
                | set(algo_phrase_hits)
            )
            ner_entity_density = ner_entity_count / token_count

            # POS-based features (lightweight noun/verb signal)
            noun_ratio, compound_noun_count, verb_ratio = self._pos_features(tokens, token_count)

            feats.append(
                [
                    domain_ratio,
                    bigram_ratio,
                    rare_idf_mean,
                    pos_domain_ratio,
                    neg_ratio,
                    is_short,
                    ner_entity_count,
                    ner_gate_count,
                    ner_unique_entities,
                    ner_entity_density,
                    noun_ratio,
                    compound_noun_count,
                    verb_ratio,
                ]
            )

        feats = np.array(feats, dtype=float)
        return feats

    def _scale_features(self, feats: np.ndarray) -> np.ndarray:
        """Min-max scale each column to [0, 1].

        Parameters
        ----------
        feats : numpy.ndarray
            Feature matrix to scale.

        Returns
        -------
        numpy.ndarray
            Scaled feature matrix.
        """
        if feats.size == 0:
            return feats

        scaled = feats.copy()
        for j in range(scaled.shape[1]):
            col = scaled[:, j]
            c_min, c_max = col.min(), col.max()
            if c_max - c_min > 0:
                scaled[:, j] = (col - c_min) / (c_max - c_min)
            else:
                scaled[:, j] = 0.0

        return scaled

    def _pos_features(self, tokens: List[str], token_count: int):
        """Compute lightweight POS features using nltk if available.

        Parameters
        ----------
        tokens : list[str]
            Tokens to tag.
        token_count : int
            Total token count (used for normalization).

        Returns
        -------
        tuple
            ``(noun_ratio, compound_noun_count, verb_ratio)``; zeros if tagging unavailable.
        """
        if not pos_tag or not tokens:
            return 0.0, 0.0, 0.0

        try:
            tagged = pos_tag(tokens)
        except Exception:
            return 0.0, 0.0, 0.0

        noun_tags = {"NN", "NNS", "NNP", "NNPS"}
        verb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

        noun_count = sum(1 for _, t in tagged if t in noun_tags)
        verb_count = sum(1 for _, t in tagged if t in verb_tags)

        # Count contiguous noun compounds (length >=2)
        compound_noun_count = 0
        run_len = 0
        for _, t in tagged:
            if t in noun_tags:
                run_len += 1
            else:
                if run_len >= 2:
                    compound_noun_count += 1
                run_len = 0
        if run_len >= 2:
            compound_noun_count += 1

        noun_ratio = noun_count / token_count
        verb_ratio = verb_count / token_count
        return noun_ratio, float(compound_noun_count), verb_ratio
    
    def filter_figures(self, figures: List[Figure]) -> List[Figure]:
        """Apply TF-IDF filtering to figures and attach scores.

        Parameters
        ----------
        figures : list[Figure]
            Figures whose captions will be evaluated.

        Returns
        -------
        list[Figure]
            Figures with similarity fields populated.
        """
        texts = [(f.caption or "") for f in figures]
        
        vectorizer = TfidfVectorizer(
            analyzer=self.preprocessor.tfidf_analyzer,
            vocabulary=self.ALLOWED_VOCAB,
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            norm='l2'
        )
        
        tfidf = vectorizer.fit_transform(texts + list(QUERY_SETS.values()))
        vocab = vectorizer.get_feature_names_out()
        idf_values = vectorizer.idf_
        idf_map = {term: idf for term, idf in zip(vocab, idf_values)}
        caption_vecs = tfidf[:len(texts)]
        query_vecs = tfidf[len(texts):]

        # Optional feature concatenation
        if USE_CUSTOM_TFIDF_FEATURES:
            CUSTOM_FEATURE_WEIGHT = 0.6  # Weight λ for custom features to prevent distorting cosine space
            custom_feats = self._compute_custom_features(texts + list(QUERY_SETS.values()), idf_map)
            weighted_feats = self._scale_features(custom_feats) * CUSTOM_FEATURE_WEIGHT
            # Split back into captions/queries
            cap_feats = weighted_feats[: len(texts)]
            qry_feats = weighted_feats[len(texts) :]
            cap_sparse = csr_matrix(cap_feats)
            qry_sparse = csr_matrix(qry_feats)
            cap_aug = hstack([caption_vecs, cap_sparse])
            qry_aug = hstack([query_vecs, qry_sparse])
        else:
            cap_aug = caption_vecs
            qry_aug = query_vecs

        sims_aug = cosine_similarity(cap_aug, qry_aug)
        sims_base = cosine_similarity(caption_vecs, query_vecs)
        
        for i, figure in enumerate(figures):
            preproc = self.preprocessor.preprocess_text_to_string(figure.caption)

            per_query = {
                name: float(sims_aug[i, j])
                for j, name in enumerate(QUERY_SETS.keys())
            }
            best_sim_aug = max(per_query.values())
            best_query = max(per_query, key=per_query.get)

            per_query_base = {
                name: float(sims_base[i, j])
                for j, name in enumerate(QUERY_SETS.keys())
            }
            best_sim_base = max(per_query_base.values())

            neg_count = self.preprocessor.count_negative_tokens(preproc)
            # Penalty as multiplicative factor: each negative token reduces score by NEGATIVE_PENALTY_ALPHA fraction
            if USE_NEGATIVE_PENALTY and neg_count > 0:
                penalty_factor = (NEGATIVE_PENALTY_ALPHA / 100.0) * neg_count  # e.g., 5% per negative token
                penalty_factor = min(penalty_factor, 0.9)  # Cap at 90% reduction
                final_sim = best_sim_aug * (1.0 - penalty_factor)
            else:
                penalty_factor = 0.0
                final_sim = best_sim_aug

            figure.preprocessed_text = preproc
            figure.similarities = per_query
            figure.best_query = best_query
            figure.similarity_raw = best_sim_base
            figure.similarity = final_sim
            figure.negative_tokens = neg_count
            figure.penalty = penalty_factor
        
        return figures
    
    def get_accepted_figures(self, figures: List[Figure]) -> List[Figure]:
        """Get figures that pass the TF-IDF threshold."""
        # Filter by threshold AND minimum token overlap
        MIN_TOKEN_OVERLAP = 2  # Require at least 2 matching tokens
        accepted = []
        for f in figures:
            if f.similarity >= SIMILARITY_THRESHOLD:
                # Count tokens that match allowed vocab (using same analyzer)
                tokens = set(self.preprocessor.tfidf_analyzer(f.caption or ""))
                overlap = len(tokens & self.ALLOWED_VOCAB)
                if overlap >= MIN_TOKEN_OVERLAP:
                    accepted.append(f)
        return accepted
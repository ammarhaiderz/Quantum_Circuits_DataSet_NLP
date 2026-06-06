"""
Text preprocessing utilities.
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config.settings import USE_STEMMING, USE_STOPWORDS, NORMALIZE_HYPHENS
from config.queries import STEMMER, PROTECTED_TOKENS, NEGATIVE_TOKENS, QUANTUM_POSITIVE_TOKENS


class TextPreprocessor:
    """Handles text preprocessing for captions."""
    
    def __init__(self):
        """Initialize the preprocessor with the configured stemmer."""
        self.stemmer = STEMMER
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by lowercasing and optionally replacing separators.

        Parameters
        ----------
        text : str
            Raw text to normalize.

        Returns
        -------
        str
            Lowercased text with ``-``, ``_``, and ``/`` replaced by spaces
            when ``NORMALIZE_HYPHENS`` is enabled.
        """
        text = text.lower()
        if NORMALIZE_HYPHENS:
            text = re.sub(r"[-_/]", " ", text)
        return text
    
    def clean_caption_text(self, text: str) -> str:
        """Clean LaTeX-heavy captions while preserving circuit vocabulary.

        Handles math removal, figure references, compound quantum terms, and
        selective hyphen/underscore normalization tailored for captions.

        Parameters
        ----------
        text : str
            Caption text potentially containing LaTeX artifacts.

        Returns
        -------
        str
            Cleaned lowercase caption text with preserved quantum terms.
        """
        text = text.lower()
        
        # Check if we should remove "circuit"
        has_quantum_term = any(term in text for term in QUANTUM_POSITIVE_TOKENS)
        
        # Remove LaTeX math
        text = re.sub(r"\$.*?\$", " ", text)
        
        # Remove figure references
        text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text)
        
        # Handle LaTeX subscripts/superscripts: H_1 -> h1, qubit_i -> qubiti
        text = re.sub(r"([a-z]+)_([a-z0-9]+)", r"\1\2", text)
        
        # Handle common quantum compound terms - preserve semantic units
        # Convert multi-qubit, two-qubit, n-qubit -> multiqubit, twoqubit, nqubit
        text = re.sub(r"\b(multi|two|three|single|n)[\-_](qubit|gate|body|level|photon)", r"\1\2", text)
        
        # Handle controlled gates: controlled-not -> controllednot, controlled-z -> controlledz
        text = re.sub(r"\bcontrolled[\-_]([a-z]+)", r"controlled\1", text)
        
        # Convert hyphenated algorithm names: deutsch-jozsa -> deutschjozsa
        text = re.sub(r"\b([a-z]+)[\-_]([a-z]+)\s+(algorithm|circuit|protocol)", r"\1\2 \3", text)
        
        # Normalize remaining hyphens/underscores/slashes to spaces
        # But preserve in specific gate names that need them
        text = re.sub(r"(?<!controlled)(?<!multi)(?<!two)(?<!three)[\-_/]+", " ", text)
        
        # Remove scientific notation
        text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
        text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)
        
        # Remove standalone numbers (but keep attached to letters like h1, qubit2)
        text = re.sub(r"\b\d+\b", " ", text)
        
        # Preserve "circuit" only if quantum context is present
        if not has_quantum_term:
            # Remove "circuit" if no quantum terms
            text = re.sub(r"\bcircuit(s)?\b", " ", text)
        
        # Normalize punctuation
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def light_clean_for_sbert(self, text: str) -> str:
        """Light preprocessing for SBERT that preserves natural language.

        Removes LaTeX math and figure references while keeping punctuation,
        capitalization, and stopwords intact.

        Parameters
        ----------
        text : str
            Raw caption text.

        Returns
        -------
        str
            Lightly cleaned text suitable for SBERT inputs.
        """
        # Remove LaTeX math
        text = re.sub(r"\$.*?\$", " ", text)
        
        # Remove figure references like "Fig. 3" or "figure 2"
        text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text, flags=re.IGNORECASE)
        
        # Keep everything else as-is for BERT to understand
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def stem_token(self, token: str) -> str:
        """Stem a token while respecting protected terms.

        Parameters
        ----------
        token : str
            Token to stem.

        Returns
        -------
        str
            Stemmed token unless it is protected or stemming is disabled.
        """
        if token in PROTECTED_TOKENS:
            return token
        return self.stemmer.stem(token) if USE_STEMMING else token
    
    def tfidf_analyzer(self, text: str) -> list:
        """Analyze text into tokens for TF-IDF vectorization.

        Parameters
        ----------
        text : str
            Raw caption text.

        Returns
        -------
        list
            List of normalized tokens ready for TF-IDF.
        """
        text = self.clean_caption_text(text)
        text = self.normalize_text(text)
        tokens = re.findall(r"[a-z0-9]+", text)
        
        if USE_STOPWORDS:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        
        if USE_STEMMING:
            tokens = [self.stem_token(t) for t in tokens]
        
        return tokens
    
    def preprocess_text_to_string(self, text: str) -> str:
        """Convert text to a whitespace-joined preprocessed string.

        Parameters
        ----------
        text : str
            Raw caption text.

        Returns
        -------
        str
            Space-joined TF-IDF tokens.
        """
        return " ".join(self.tfidf_analyzer(text))
    
    def count_negative_tokens(self, preprocessed_text: str) -> int:
        """Count occurrences of negative tokens in preprocessed text.

        Parameters
        ----------
        preprocessed_text : str
            Whitespace-separated token string.

        Returns
        -------
        int
            Number of tokens present in ``NEGATIVE_TOKENS``.
        """
        tokens = preprocessed_text.split()
        return sum(t in NEGATIVE_TOKENS for t in tokens)
    
    def preprocess_filename(self, name: str) -> list:
        """Preprocess a filename into tokens for negative checking.

        Parameters
        ----------
        name : str
            Filename to tokenize.

        Returns
        -------
        list
            Normalized (and optionally stemmed) filename tokens.
        """
        name = name.lower()
        name = re.sub(r"[^a-z0-9]+", " ", name)
        tokens = name.split()
        return [self.stemmer.stem(t) if USE_STEMMING else t for t in tokens]
    

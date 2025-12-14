"""
Text preprocessing utilities for quantum circuit extraction.
"""
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

from config import (
    USE_STEMMING, USE_STOPWORDS, NORMALIZE_HYPHENS,
    PROTECTED_TOKENS, QUANTUM_POSITIVE_TOKENS,
    NEGATIVE_RAW_TOKENS, FILENAME_NEGATIVE_RAW,
    HARD_REJECT_CATEGORIES, SOFT_PENALTY_WORDS, HARD_REJECT_PHRASES
)

STEMMER = PorterStemmer()

# Precompute stemmed negative tokens for efficiency
NEGATIVE_TOKENS = {
    STEMMER.stem(token) for token in NEGATIVE_RAW_TOKENS
}

FILENAME_NEGATIVE_TOKENS = {
    STEMMER.stem(t) for t in FILENAME_NEGATIVE_RAW
}

# Regular expressions
FIG_RE = re.compile(r"\\begin{figure}.*?\\end{figure}", re.DOTALL)
CAP_RE = re.compile(r"\\caption\{([^}]*)\}", re.DOTALL)
IMG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", re.DOTALL)


class TextPreprocessor:
    """Handles all text preprocessing operations."""
    
    def __init__(self):
        self.stepper = PorterStemmer()
    
    def clean_caption_text(self, text: str) -> str:
        """
        Cleans LaTeX-heavy captions without destroying circuit vocabulary.
        REMOVES 'circuit' ONLY if no quantum terms are present.
        """
        text = text.lower()

        # Check if we should remove "circuit"
        has_quantum_term = any(
            term in text for term in QUANTUM_POSITIVE_TOKENS
        )
        
        # Remove LaTeX math
        text = re.sub(r"\$.*?\$", " ", text)

        # Remove figure references
        text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text)

        # Remove scientific notation
        text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
        text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)

        # Remove standalone numbers
        text = re.sub(r"\b\d+\b", " ", text)

        # NORMALIZE "circuit" word based on quantum context
        if not has_quantum_term:
            # Remove "circuit" if no quantum terms
            text = re.sub(r"\bcircuit(s)?\b", " ", text)
        else:
            # Keep "circuit" but normalize it (optional)
            text = re.sub(r"\bcircuit(s)?\b", " quantum_circuit ", text)

        # Normalize punctuation
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    def normalize_text(self, text):
        text = text.lower()
        if NORMALIZE_HYPHENS:
            text = re.sub(r"[-_/]", " ", text)
        return text
    
    def stem_token(self, token):
        if token in PROTECTED_TOKENS:
            return token
        return self.stepper.stem(token)
    
    def tfidf_analyzer(self, text):
        """Tokenizer for TF-IDF."""
        text = self.clean_caption_text(text)
        text = self.normalize_text(text)
        tokens = re.findall(r"[a-z0-9]+", text)

        if USE_STOPWORDS:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

        if USE_STEMMING:
            tokens = [self.stem_token(t) for t in tokens]

        return tokens
    
    def preprocess_text_to_string(self, text):
        return " ".join(self.tfidf_analyzer(text))
    
    def count_negative_tokens(self, preprocessed_text):
        tokens = preprocessed_text.split()
        return sum(t in NEGATIVE_TOKENS for t in tokens)
    
    def preprocess_filename(self, name):
        name = name.lower()
        name = re.sub(r"[^a-z0-9]+", " ", name)
        tokens = name.split()
        return [self.stepper.stem(t) for t in tokens]
    
    def filename_is_negative(self, img_path):
        fname = os.path.basename(img_path)
        tokens = self.preprocess_filename(fname)
        return any(t in FILENAME_NEGATIVE_TOKENS for t in tokens)

    def light_clean_for_sbert(self, text: str) -> str:
        """
        Light preprocessing for SBERT that preserves natural language.
        Removes LaTeX, numbers, citations but preserves natural flow.
        Keeps vocabulary like 'circuit', 'qubit', 'gate' intact.
        """
        text = text.lower()

        # Remove LaTeX math mode
        text = re.sub(r"\$.*?\$", " ", text)

        # Remove LaTeX commands like \textbf{...}, \emph{...}
        text = re.sub(r"\\[a-z]+\{", " ", text)
        text = re.sub(r"\}", " ", text)

        # Remove figure references
        text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text)

        # Remove citations: \cite{...} or [12]
        text = re.sub(r"\\cite\{[^}]*\}", " ", text)
        text = re.sub(r"\[\d+\]", " ", text)

        # Remove standalone numbers but keep compound words (e.g., '2qubit')
        text = re.sub(r"\b\d+\b", " ", text)
        text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
        text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)

        # Normalize punctuation to spaces
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def check_hard_rejection(self, text: str) -> bool:
        """
        Check if caption contains clear non-quantum indicators.
        Returns True if should be HARD REJECTED.
        """
        text_lower = text.lower()

        # Check for hard-reject exact phrases
        for phrase in HARD_REJECT_PHRASES:
            if phrase in text_lower:
                return True

        # Check for category-level hard rejection
        # (e.g., "electrical" + "voltage" + "circuit")
        for category, keywords in HARD_REJECT_CATEGORIES.items():
            keywords_found = [
                kw for kw in keywords if kw in text_lower
            ]
            # If multiple keywords from same category + no quantum terms
            if len(keywords_found) >= 2:
                has_quantum = any(
                    q in text_lower
                    for q in QUANTUM_POSITIVE_TOKENS
                )
                if not has_quantum:
                    return True

        return False

    def calculate_context_aware_penalty(self, text: str) -> float:
        """
        Calculate penalty based on negative words and their context.
        Considers word proximity to quantum terms to avoid false positives.
        
        Returns penalty as float (positive = reduction, 0 = no penalty).
        """
        text_lower = text.lower()
        total_penalty = 0.0

        # Check each negative word and its context
        for word, weight in SOFT_PENALTY_WORDS.items():
            if word in text_lower:
                # Look for context: is this word near a quantum term?
                # Extract window around the negative word
                idx = text_lower.find(word)
                if idx >= 0:
                    window_start = max(0, idx - 50)
                    window_end = min(len(text_lower), idx + 50)
                    window = text_lower[window_start:window_end]

                    # Check if quantum terms are in the window
                    nearby_quantum = any(
                        q in window for q in QUANTUM_POSITIVE_TOKENS
                    )

                    # Reduce penalty if quantum term is nearby
                    # (context suggests hybrid/measurement context)
                    if nearby_quantum:
                        penalty_reduction = 0.5  # Reduce penalty by 50%
                        total_penalty += weight * penalty_reduction
                    else:
                        total_penalty += weight

        return abs(total_penalty)  # Return as positive value


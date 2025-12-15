"""
Text preprocessing utilities.
"""

import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config.settings import USE_STEMMING, USE_STOPWORDS, NORMALIZE_HYPHENS
from config.queries import STEMMER, PROTECTED_TOKENS

class TextPreprocessor:
    """Handles text preprocessing for captions."""
    
    def __init__(self):
        self.stemmer = STEMMER
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by lowercasing and handling hyphens."""
        text = text.lower()
        if NORMALIZE_HYPHENS:
            text = re.sub(r"[-_/]", " ", text)
        return text
    
    def clean_caption_text(self, text: str) -> str:
        """
        Cleans LaTeX-heavy captions without destroying circuit vocabulary.
        """
        if not text:
            return ""
        
        text = text.lower()
        
        # Remove LaTeX math
        text = re.sub(r"\$.*?\$", " ", text)
        
        # Remove figure references
        text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text, flags=re.IGNORECASE)
        
        # Handle LaTeX subscripts/superscripts
        text = re.sub(r"([a-z]+)_([a-z0-9]+)", r"\1\2", text)
        
        # Handle common quantum compound terms
        text = re.sub(r"\b(multi|two|three|single|n)[\-_](qubit|gate|body|level)", r"\1\2", text)
        text = re.sub(r"\bcontrolled[\-_]([a-z]+)", r"controlled\1", text)
        text = re.sub(r"\b([a-z]+)[\-_]([a-z]+)\s+(algorithm|circuit|protocol)", r"\1\2 \3", text)
        
        # Normalize remaining separators
        text = re.sub(r"(?<!controlled)(?<!multi)(?<!two)(?<!three)[\-_/]+", " ", text)
        
        # Remove scientific notation and standalone numbers
        text = re.sub(r"\b\d+(\.\d+)?e[-+]?\d+\b", " ", text)
        text = re.sub(r"\b10\^\{?-?\d+\}?\b", " ", text)
        text = re.sub(r"\b\d+\b", " ", text)
        
        # Normalize punctuation and whitespace
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def light_clean_for_sbert(self, text: str) -> str:
        """
        Light preprocessing for SBERT that preserves natural language.
        """
        if not text:
            return ""
        
        # Remove LaTeX math
        text = re.sub(r"\$.*?\$", " ", text)
        
        # Remove figure references
        text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def stem_token(self, token: str) -> str:
        """Stem token while protecting certain tokens."""
        if token in PROTECTED_TOKENS:
            return token
        return self.stemmer.stem(token) if USE_STEMMING else token
    
    def tfidf_analyzer(self, text: str) -> list:
        """Analyze text for TF-IDF vectorization."""
        if not text:
            return []
        
        text = self.clean_caption_text(text)
        text = self.normalize_text(text)
        tokens = re.findall(r"[a-z0-9]+", text)
        
        if USE_STOPWORDS:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        
        if USE_STEMMING:
            tokens = [self.stem_token(t) for t in tokens]
        
        return tokens
    
    def preprocess_text_to_string(self, text: str) -> str:
        """Convert text to preprocessed string."""
        tokens = self.tfidf_analyzer(text)
        return " ".join(tokens)
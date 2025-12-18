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
        Handles infixes (hyphens, underscores, slashes) intelligently.
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
        """
        Light preprocessing for SBERT that preserves natural language.
        BERT models benefit from:
        - Punctuation (semantic markers)
        - Capitalization (some context)
        - Grammar (BERT understands morphology)
        - Stopwords (context words)
        
        ONLY removes LaTeX noise and normalizes whitespace.
        """
        # Remove LaTeX math
        text = re.sub(r"\$.*?\$", " ", text)
        
        # Remove figure references like "Fig. 3" or "figure 2"
        text = re.sub(r"\b(fig\.?|figure)\s*\d+\b", " ", text, flags=re.IGNORECASE)
        
        # Optionally: normalize unicode symbols to spaces
        # Keep everything else as-is for BERT to understand
        
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
        return " ".join(self.tfidf_analyzer(text))
    
    def count_negative_tokens(self, preprocessed_text: str) -> int:
        """Count negative tokens in preprocessed text."""
        tokens = preprocessed_text.split()
        return sum(t in NEGATIVE_TOKENS for t in tokens)
    
    def preprocess_filename(self, name: str) -> list:
        """Preprocess filename for negative checking."""
        name = name.lower()
        name = re.sub(r"[^a-z0-9]+", " ", name)
        tokens = name.split()
        return [self.stemmer.stem(t) if USE_STEMMING else t for t in tokens]
    

if __name__ == "__main__":
    tp = TextPreprocessor()

    test_captions = [
        "FIG. 1. Schematic diagrams of (a) discrete-variable and (b) continuous-variable quantum teleportation.",
        """FIG. 1. Experimental setup with the DC circuit on the
left in purple and the reflectometry circuit on the right in
green. The Keysight chassis handles control and readout of
the SiGe quantum dot device. On the quantum dot (QD)
device schematic, red gates act as barrier gates, green gates
as plunger gates, blue gates as reservoirs and purple gates as
confinement gates. A voltage divider is used to increase the
resolution of the M3201A’s voltage applied to the ohmic con-
tacts by a factor of 100""",
        "A list of topics to categorize the field of quantum machine learning and its algorithms.",
        r"Fig. 5: Circuit $U(\theta)$ acting on qubit$_i$ with H_1 and R_z(\phi).",
        "A typical quantum circuit synthesis flow.",
        "Equivalent electrical circuit representation of the resonator system.",
    ]

    print("=" * 80)
    print(" QUICK TEXT PREPROCESSING CHECK ")
    print("=" * 80)

    for i, caption in enumerate(test_captions, 1):
        print(f"\n--- Case {i} ---")
        print("ORIGINAL:")
        print(caption)

        tfidf_tokens = tp.tfidf_analyzer(caption)
        tfidf_text = tp.preprocess_text_to_string(caption)
        neg_count = tp.count_negative_tokens(tfidf_text)
        sbert_text = tp.light_clean_for_sbert(caption)

        print("\nTF-IDF TOKENS:")
        print(tfidf_tokens)

        print("\nTF-IDF STRING:")
        print(tfidf_text)

        print(f"\nNEGATIVE TOKEN COUNT: {neg_count}")

        print("\nSBERT INPUT:")
        print(sbert_text)

        print("-" * 80)
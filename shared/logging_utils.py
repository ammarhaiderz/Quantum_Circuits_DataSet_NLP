"""
Logging and debugging utilities.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from config.settings import LOG_LEVEL, ENABLE_DEBUG_PRINTS, SAVE_INTERMEDIATE_RESULTS
import sys
from logging import Formatter


class SafeFormatter(Formatter):
    """Formatter that replaces characters not encodable by the stream encoding.

    This prevents UnicodeEncodeError when logging to consoles with limited
    encodings (e.g., cp1252 on Windows). It emits a sanitized string where
    unencodable characters are replaced.
    """
    def __init__(self, fmt=None, datefmt=None, stream_encoding=None):
        """Initialize formatter with optional stream encoding.

        Parameters
        ----------
        fmt : str, optional
            Log format string.
        datefmt : str, optional
            Date format string.
        stream_encoding : str, optional
            Encoding to use when sanitizing output; defaults to stream encoding.
        """
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.stream_encoding = stream_encoding

    def format(self, record):
        """Format a log record while replacing unencodable characters."""
        s = super().format(record)
        enc = self.stream_encoding or getattr(sys.stderr, 'encoding', None) or 'utf-8'
        try:
            # encode+decode with replace to ensure the string is safe for the stream
            return s.encode(enc, errors='replace').decode(enc)
        except Exception:
            # fallback: replace non-ascii
            return s.encode('utf-8', errors='replace').decode('utf-8')


class Logger:
    """Custom logger for the pipeline."""
    
    def __init__(self, name: str = "QuantumExtractor"):
        """Create and configure a logger instance.

        Parameters
        ----------
        name : str, optional
            Logger name (default ``"QuantumExtractor"``).
        """
        self.logger = logging.getLogger(name)
        self.setup_logging()
        
    def setup_logging(self):
        """Set up console (and optional file) handlers and formatters."""
        log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        # avoid adding multiple handlers if setup_logging called repeatedly
        if self.logger.handlers:
            for h in list(self.logger.handlers):
                self.logger.removeHandler(h)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = SafeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            stream_encoding=getattr(console_handler.stream, 'encoding', None)
        )
        console_handler.setFormatter(console_formatter)

        self.logger.setLevel(log_level)
        self.logger.addHandler(console_handler)
        
        # File handler for intermediate results
        if SAVE_INTERMEDIATE_RESULTS:
            file_handler = logging.FileHandler(
                f"logs/extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = SafeFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                stream_encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str):
        """Emit a debug log when debug printing is enabled."""
        if ENABLE_DEBUG_PRINTS:
            self.logger.debug(msg)
    
    def info(self, msg: str):
        """Emit an info log."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Emit a warning log."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Emit an error log."""
        self.logger.error(msg)
    
    def print_debug_info(self, paper_id: str, figures: List, top_n: int = 5):
        """Print top-N TF-IDF caption debug info for a paper.

        Parameters
        ----------
        paper_id : str
            Identifier of the paper being processed.
        figures : list
            Figures with TF-IDF scores and captions.
        top_n : int, optional
            Number of top captions to show (default ``5``).
        """
        if not ENABLE_DEBUG_PRINTS:
            return
        
        self.info(f"\n Processing {paper_id}")
        
        if figures:
            sorted_figures = sorted(figures, key=lambda x: x.similarity, reverse=True)
            
            self.debug(f"\n    Top {top_n} captions by TF-IDF:")
            for i, f in enumerate(sorted_figures[:top_n], start=1):
                self.debug(
                    f"\n   [{i}] tfidf={f.similarity:.4f} "
                    f"(raw={f.similarity_raw:.4f}, neg={f.negative_tokens}, pen={f.penalty:.4f})"
                )
                self.debug("   RAW:")
                self.debug("   " + f.caption[:100] + ("..." if len(f.caption) > 100 else ""))
                self.debug("   PREPROCESSED:")
                self.debug("   " + f.preprocessed_text)
    
    def print_statistics(self, stats: Dict[str, Any], df=None):
        """Print high-level pipeline statistics and optional detailed metrics.

        Parameters
        ----------
        stats : dict
            Aggregated counters from the pipeline.
        df : pandas.DataFrame, optional
            DataFrame with figure scores for detailed breakdowns.
        """
        self.info("\n" + "="*40 + " SUMMARY " + "="*40)
        self.info(f"Papers checked: {stats['papers_checked']}")
        self.info(f"Papers with figures: {stats['papers_with_figures']}")
        self.info(f"Papers with candidates: {stats['papers_with_candidates']}")
        self.info(f"Papers with extracted images: {stats['papers_with_extracted']}")
        self.info(f"Total figures seen: {stats['total_figures_seen']}")
        self.info(f"Total images saved: {stats['total_saved']}")
        
        if df is not None and not df.empty:
            self._print_detailed_statistics(df)
        
        self.info("="*88)
    
    def _print_detailed_statistics(self, df):
        """Print detailed TF-IDF, SBERT, and combined score statistics."""
        from config.settings import SIMILARITY_THRESHOLD, SBERT_MIN_SIM
        
        self.info("\n" + "="*88)
        self.info(" TF-IDF STATISTICS (for tuning SIMILARITY_THRESHOLD)")
        self.info("="*88)
        
        # TF-IDF stats
        tfidf_scores = df['similarity']
        self.info(f"Total figures: {len(df)}")
        self.info(f"TF-IDF Mean: {tfidf_scores.mean():.4f} | "
                  f"Median: {tfidf_scores.median():.4f} | "
                  f"Std: {tfidf_scores.std():.4f}")
        self.info(f"TF-IDF Min: {tfidf_scores.min():.4f} | "
                  f"Max: {tfidf_scores.max():.4f}")
        
        # TF-IDF percentiles
        self.info("\nTF-IDF Percentiles:")
        for p in [25, 50, 75, 90, 95, 99]:
            val = tfidf_scores.quantile(p/100)
            self.info(f"  {p}th: {val:.4f}")
        
        # Distribution by threshold
        self.info("\nTF-IDF Distribution (figures passing threshold):")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            count = (tfidf_scores >= threshold).sum()
            pct = (count / len(df)) * 100
            marker = " <- CURRENT" if abs(threshold - SIMILARITY_THRESHOLD) < 0.01 else ""
            self.info(f"  >={threshold:.1f}: {count:4d} ({pct:5.1f}%){marker}")
        
        # SBERT stats (if available)
        if "sbert_sim" in df.columns:
            sbert_scores = df['sbert_sim'].dropna()
            
            if len(sbert_scores) > 0:
                self.info("\n" + "="*88)
                self.info(" SBERT STATISTICS (for tuning SBERT_MIN_SIM)")
                self.info("="*88)
                
                self.info(f"Figures with SBERT scores: {len(sbert_scores)}")
                self.info(f"SBERT Mean: {sbert_scores.mean():.4f} | "
                          f"Median: {sbert_scores.median():.4f} | "
                          f"Std: {sbert_scores.std():.4f}")
                self.info(f"SBERT Min: {sbert_scores.min():.4f} | "
                          f"Max: {sbert_scores.max():.4f}")
                
                # SBERT percentiles
                self.info("\nSBERT Percentiles:")
                for p in [25, 50, 75, 90, 95, 99]:
                    val = sbert_scores.quantile(p/100)
                    self.info(f"  {p}th: {val:.4f}")
                
                # SBERT distribution
                self.info("\nSBERT Distribution (figures passing threshold):")
                for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                    count = (sbert_scores >= threshold).sum()
                    pct = (count / len(sbert_scores)) * 100
                    marker = " ← CURRENT" if abs(threshold - SBERT_MIN_SIM) < 0.01 else ""
                    self.info(f"  >={threshold:.1f}: {count:4d} ({pct:5.1f}%){marker}")
                
                # Combined score stats (if available)
                if "combined_score" in df.columns:
                    combined_scores = df['combined_score'].dropna()
                    if len(combined_scores) > 0:
                        from config.settings import USE_COMBINED_SCORE, TFIDF_WEIGHT, SBERT_WEIGHT, COMBINED_THRESHOLD
                        
                        self.info("\n" + "="*88)
                        self.info(" COMBINED SCORE STATISTICS (for tuning weights)")
                        self.info("="*88)
                        
                        mode = "ENABLED" if USE_COMBINED_SCORE else "DISABLED (using cascade)"
                        self.info(f"Mode: {mode}")
                        if USE_COMBINED_SCORE:
                            self.info(f"Weights: TF-IDF={TFIDF_WEIGHT:.2f}, SBERT={SBERT_WEIGHT:.2f}")
                        
                        self.info(f"Combined Mean: {combined_scores.mean():.4f} | "
                                  f"Median: {combined_scores.median():.4f} | "
                                  f"Std: {combined_scores.std():.4f}")
                        self.info(f"Combined Min: {combined_scores.min():.4f} | "
                                  f"Max: {combined_scores.max():.4f}")
                        
                        # Combined percentiles
                        self.info("\nCombined Score Percentiles:")
                        for p in [25, 50, 75, 90, 95, 99]:
                            val = combined_scores.quantile(p/100)
                            self.info(f"  {p}th: {val:.4f}")
                        
                        # Combined distribution
                        self.info("\nCombined Score Distribution:")
                        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                            count = (combined_scores >= threshold).sum()
                            pct = (count / len(combined_scores)) * 100
                            marker = " <- CURRENT" if abs(threshold - COMBINED_THRESHOLD) < 0.01 else ""
                            self.info(f"  >={threshold:.1f}: {count:4d} ({pct:5.1f}%){marker}")
                
                # Selection funnel
                self.info("\n" + "="*88)
                self.info(" SELECTION FUNNEL")
                self.info("="*88)
                selected = df['selected'].sum() if 'selected' in df.columns else 0
                extracted = df['extracted'].sum() if 'extracted' in df.columns else 0
                
                from config.settings import USE_COMBINED_SCORE, COMBINED_THRESHOLD
                
                self.info(f"Total figures seen:      {len(df):4d} (100.0%)")
                tfidf_passed = (df['similarity'] >= SIMILARITY_THRESHOLD).sum()
                self.info(f"Passed TF-IDF gate:      {tfidf_passed:4d} "
                          f"({(tfidf_passed/len(df)*100):5.1f}%)")
                
                # Token overlap info if available
                if 'token_overlap_count' in df.columns:
                    overlap_passed = (df['token_overlap_count'] >= 2).sum()
                    self.info(f"Passed Token overlap:    {overlap_passed:4d} "
                              f"({(overlap_passed/len(df)*100):5.1f}%)  (>=2 tokens)")
                
                if USE_COMBINED_SCORE and 'combined_score' in df.columns:
                    combined_passed = (df['combined_score'] >= COMBINED_THRESHOLD).sum()
                    self.info(f"Passed Combined gate:    {combined_passed:4d} "
                              f"({(combined_passed/len(df)*100):5.1f}%)")
                elif len(sbert_scores) > 0:
                    sbert_passed = (sbert_scores >= SBERT_MIN_SIM).sum()
                    self.info(f"Passed SBERT gate:       {sbert_passed:4d} "
                              f"({(sbert_passed/len(sbert_scores)*100):5.1f}%)")
                
                self.info(f"Selected for extraction: {selected:4d} "
                          f"({(selected/len(df)*100):5.1f}%)")
                self.info(f"Successfully extracted:  {extracted:4d} "
                          f"({(extracted/len(df)*100):5.1f}%)")
            else:
                self.warning("\n No SBERT scores available")

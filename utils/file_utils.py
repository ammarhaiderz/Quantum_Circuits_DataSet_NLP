"""
File utility functions.
"""

import os
import pandas as pd
from typing import List, Optional
from config.settings import OUTPUT_DIR, SUPPORTED_EXT


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def read_arxiv_ids(filename: str) -> List[str]:
        """Read arXiv IDs from a text file.

        Parameters
        ----------
        filename : str
            Path to the file containing arXiv IDs (one per line).

        Returns
        -------
        list[str]
            List of IDs with any leading ``arXiv:`` prefix removed.
        """
        with open(filename, "r") as f:
            return [l.strip().replace("arXiv:", "") for l in f if l.strip()]
    
    @staticmethod
    def clear_output_dir(extensions: Optional[List[str]] = None):
        """Clear previous images from the output directory.

        Parameters
        ----------
        extensions : list[str], optional
            File extensions to delete; defaults to ``SUPPORTED_EXT``.
        """
        if extensions is None:
            extensions = SUPPORTED_EXT
        
        removed_count = 0
        for fname in os.listdir(OUTPUT_DIR):
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.isfile(fpath) and any(fname.lower().endswith(ext) for ext in extensions):
                os.remove(fpath)
                removed_count += 1
        
        if removed_count > 0:
            print(f"🧹 Removed {removed_count} previous images")
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filename: str = "caption_text_log.csv"):
        """Save a DataFrame to CSV in the output directory.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to persist.
        filename : str, optional
            Target CSV filename (default ``'caption_text_log.csv'``).

        Returns
        -------
        str
            Path to the saved CSV file.
        """
        df_path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(df_path, index=False)
        print(f"💾 Saved caption log to: {df_path}")
        
        # Save summary statistics CSV
        FileUtils.save_statistics_summary(df)
        
        return df_path
    
    @staticmethod
    def save_statistics_summary(df: pd.DataFrame):
        """Save statistics summaries and threshold analyses.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing at least ``similarity`` (and optionally ``sbert_sim``).
        """
        summary_path = os.path.join(OUTPUT_DIR, "statistics_summary.csv")
        
        summary_data = []
        
        # TF-IDF statistics
        tfidf_scores = df['similarity']
        summary_data.append({
            'Metric': 'TF-IDF',
            'Count': len(tfidf_scores),
            'Mean': tfidf_scores.mean(),
            'Median': tfidf_scores.median(),
            'Std': tfidf_scores.std(),
            'Min': tfidf_scores.min(),
            'Max': tfidf_scores.max(),
            'P25': tfidf_scores.quantile(0.25),
            'P50': tfidf_scores.quantile(0.50),
            'P75': tfidf_scores.quantile(0.75),
            'P90': tfidf_scores.quantile(0.90),
            'P95': tfidf_scores.quantile(0.95),
            'P99': tfidf_scores.quantile(0.99)
        })
        
        # SBERT statistics (if available)
        if 'sbert_sim' in df.columns:
            sbert_scores = df['sbert_sim'].dropna()
            if len(sbert_scores) > 0:
                summary_data.append({
                    'Metric': 'SBERT',
                    'Count': len(sbert_scores),
                    'Mean': sbert_scores.mean(),
                    'Median': sbert_scores.median(),
                    'Std': sbert_scores.std(),
                    'Min': sbert_scores.min(),
                    'Max': sbert_scores.max(),
                    'P25': sbert_scores.quantile(0.25),
                    'P50': sbert_scores.quantile(0.50),
                    'P75': sbert_scores.quantile(0.75),
                    'P90': sbert_scores.quantile(0.90),
                    'P95': sbert_scores.quantile(0.95),
                    'P99': sbert_scores.quantile(0.99)
                })
        
        # Threshold analysis - TF-IDF
        threshold_data = []
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            count = (tfidf_scores >= threshold).sum()
            threshold_data.append({
                'Gate': 'TF-IDF',
                'Threshold': threshold,
                'Count_Passing': count,
                'Percentage': (count / len(tfidf_scores)) * 100
            })
        
        # Threshold analysis - SBERT
        if 'sbert_sim' in df.columns:
            sbert_scores = df['sbert_sim'].dropna()
            if len(sbert_scores) > 0:
                for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    count = (sbert_scores >= threshold).sum()
                    threshold_data.append({
                        'Gate': 'SBERT',
                        'Threshold': threshold,
                        'Count_Passing': count,
                        'Percentage': (count / len(sbert_scores)) * 100
                    })
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"[STATS] Saved statistics summary to: {summary_path}")
        
        # Save threshold analysis
        threshold_path = os.path.join(OUTPUT_DIR, "threshold_analysis.csv")
        threshold_df = pd.DataFrame(threshold_data)
        threshold_df.to_csv(threshold_path, index=False, float_format='%.4f')
        print(f"[STATS] Saved threshold analysis to: {threshold_path}")
    
    @staticmethod
    def get_safe_filename(paper_id: str, idx: int, original_name: str) -> str:
        """Generate a safe filename for an extracted image.

        Parameters
        ----------
        paper_id : str
            Paper identifier.
        idx : int
            Sequential index for the file.
        original_name : str
            Original filename to preserve basename context.

        Returns
        -------
        str
            Sanitized filename combining paper ID, index, and original basename.
        """
        safe_pid = paper_id.replace("/", "_").replace(".", "_")
        return f"{safe_pid}_{idx}_{os.path.basename(original_name)}"
    
    @staticmethod
    def create_directory(path: str):
        """Create a directory if it does not exist.

        Parameters
        ----------
        path : str
            Directory path to create.
        """
        os.makedirs(path, exist_ok=True)
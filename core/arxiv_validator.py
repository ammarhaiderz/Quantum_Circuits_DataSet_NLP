"""
arXiv paper category filter with caching.
"""

import requests
import json
import os
import time
from typing import Optional
from config.settings import REQUEST_DELAY


class ArxivFilter:
    """Filter arXiv papers by category with caching."""
    
    def __init__(self, cache_file: str = "arxiv_category_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")
    
    def is_quantum_paper(self, arxiv_id: str) -> bool:
        """
        Check if a paper is in quant-ph category.
        Uses cache to avoid repeated API calls.
        """
        # Check cache first
        if arxiv_id in self.cache:
            return self.cache[arxiv_id]
        
        # API call
        is_quantum = self._check_arxiv_api(arxiv_id)
        
        # Cache result
        self.cache[arxiv_id] = is_quantum
        self._save_cache()
        
        return is_quantum
    
    def _check_arxiv_api(self, arxiv_id: str) -> bool:
        """Check arXiv API for paper category."""
        # Remove version suffix
        clean_id = arxiv_id.split('v')[0]
        
        try:
            # Respect rate limiting
            time.sleep(REQUEST_DELAY / 2)  # Half the download delay
            
            url = f"http://export.arxiv.org/api/query?id_list={clean_id}&max_results=1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Simple text check - much faster than XML parsing
                return 'quant-ph' in response.text
            else:
                print(f"[WARN] API returned {response.status_code} for {arxiv_id}")
                return True  # Assume quantum if API fails
                
        except Exception as e:
            print(f"[WARN] API check failed for {arxiv_id}: {e}")
            return True  # Conservative: assume quantum if check fails
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = len(self.cache)
        quantum = sum(1 for v in self.cache.values() if v)
        non_quantum = total - quantum
        
        return {
            "total": total,
            "quantum": quantum,
            "non_quantum": non_quantum,
            "quantum_percentage": (quantum / total * 100) if total > 0 else 0
        }
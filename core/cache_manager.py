"""
Cache management for arXiv downloads.
"""

import os
import time
import requests
import io
from typing import Optional
from config.settings import CACHE_DIR, REQUEST_DELAY


class CacheManager:
    """Manages caching of arXiv source files."""
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cached_source(self, arxiv_id: str) -> Optional[io.BytesIO]:
        """Get cached arXiv source if available."""
        cache_path = os.path.join(self.cache_dir, f"{arxiv_id}.tar.gz")
        
        if os.path.exists(cache_path):
            print(f"📦 Using cached source for {arxiv_id}")
            try:
                with open(cache_path, "rb") as f:
                    return io.BytesIO(f.read())
            except Exception as e:
                print(f"⚠️ Failed to read cache for {arxiv_id}: {e}")
        
        return None
    
    def download_source(self, arxiv_id: str) -> Optional[io.BytesIO]:
        """Download arXiv source with rate limiting."""
        print(f"\n📥 Downloading {arxiv_id}")
        time.sleep(REQUEST_DELAY)
        
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        cache_path = os.path.join(self.cache_dir, f"{arxiv_id}.tar.gz")
        
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                # Save to cache
                with open(cache_path, "wb") as f:
                    f.write(r.content)
                return io.BytesIO(r.content)
            else:
                print(f"❌ Download failed with status {r.status_code}")
        except Exception as e:
            print(f"❌ Download error: {e}")
        
        return None
    
    def get_source(self, arxiv_id: str) -> Optional[io.BytesIO]:
        """Get source from cache or download."""
        # Try cache first
        cached = self.get_cached_source(arxiv_id)
        if cached:
            return cached
        
        # Download if not cached
        return self.download_source(arxiv_id)
    
    def clear_cache(self):
        """Clear all cached files."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            print(f"🧹 Cleared cache directory: {self.cache_dir}")
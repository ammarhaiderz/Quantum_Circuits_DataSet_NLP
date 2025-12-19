"""
Test arXiv API to see what metadata is available.
"""

import requests

def fetch_arxiv_paper(arxiv_id: str):
    """Fetch paper metadata from arXiv API."""
    try:
        clean_id = arxiv_id.split('v')[0]
        url = f"http://export.arxiv.org/api/query?id_list={clean_id}&max_results=1"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"API error: {response.status_code}")
            return None
        
        print("API response received successfully.")
        print(response.text)
        return response.text
        
    except Exception as e:
        print(f"Error fetching arXiv paper: {e}")

if __name__ == "__main__":
    test_id = "2504.11846"
    fetch_arxiv_paper(test_id)
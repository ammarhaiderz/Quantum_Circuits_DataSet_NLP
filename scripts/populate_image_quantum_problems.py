"""
Standalone script to populate quantum_problem field in images.json.

This script is completely isolated from the circuit rendering pipeline.
It reads images.json, classifies each image's quantum problem using SBERT,
and writes back the updated images.json with quantum_problem populated.

Usage:
    python scripts/populate_image_quantum_problems.py

This script:
1. Loads its own SBERT model instance (separate from circuit pipeline)
2. Reads data/images.json
3. For each image, classifies quantum problem using its descriptions
4. Updates the quantum_problem field
5. Writes back to data/images.json

Dependencies:
- sentence_transformers
- config.quantum_problem_labels
- core.quantum_problem_classifier
"""

import json
from pathlib import Path
from typing import Dict, Any

from sentence_transformers import SentenceTransformer

from config.quantum_problem_labels import QUANTUM_PROBLEM_LABELS
from core.quantum_problem_classifier import (
    prepare_label_embeddings,
    classify_quantum_problem,
    DEFAULT_THRESHOLD,
)


# Paths
DATA_DIR = Path('data')
IMAGES_JSON_PATH = DATA_DIR / 'images.json'

# Model configuration
SBERT_MODEL_NAME = 'allenai-specter'  # Same as used in circuit pipeline


def load_images_json() -> Dict[str, Any]:
    """Load images.json file.
    
    Returns
    -------
    dict
        Dictionary mapping image filenames to metadata records.
        Returns empty dict if file doesn't exist or is invalid.
    """
    if not IMAGES_JSON_PATH.exists():
        print(f"[ERROR] {IMAGES_JSON_PATH} does not exist")
        return {}
    
    try:
        with open(IMAGES_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            print(f"[ERROR] {IMAGES_JSON_PATH} is not a valid dict structure")
            return {}
        
        return data
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse {IMAGES_JSON_PATH}: {e}")
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to load {IMAGES_JSON_PATH}: {e}")
        return {}


def save_images_json(data: Dict[str, Any]) -> bool:
    """Save updated images.json file.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping image filenames to metadata records.
    
    Returns
    -------
    bool
        True if save succeeded, False otherwise.
    """
    try:
        with open(IMAGES_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {IMAGES_JSON_PATH}: {e}")
        return False


def populate_quantum_problems(
    images_data: Dict[str, Any],
    model: SentenceTransformer,
    label_keys: list,
    label_embeddings,
    threshold: float = DEFAULT_THRESHOLD,
) -> int:
    """Populate quantum_problem field for all images.
    
    Parameters
    ----------
    images_data : dict
        Dictionary of image metadata records (mutated in-place).
    model : SentenceTransformer
        SBERT model for encoding descriptions.
    label_keys : list
        Ordered label keys from QUANTUM_PROBLEM_LABELS.
    label_embeddings : array-like
        Precomputed, normalized embeddings of label texts.
    threshold : float, optional
        Minimum cosine similarity to accept a label (default 0.45).
    
    Returns
    -------
    int
        Number of images successfully classified.
    """
    classified_count = 0
    total_images = len(images_data)
    
    print(f"\n[INFO] Processing {total_images} images...")
    
    for img_name, metadata in images_data.items():
        if not isinstance(metadata, dict):
            continue
        
        # Get descriptions for this image
        descriptions = metadata.get("description", [])
        
        if not descriptions:
            # No descriptions available - mark as unspecified
            metadata["quantum_problem"] = "Unspecified quantum circuit"
            continue
        
        # Ensure descriptions is a list of strings
        if not isinstance(descriptions, list):
            descriptions = [str(descriptions)]
        
        # Classify quantum problem
        try:
            problem_type = classify_quantum_problem(
                model,
                descriptions,
                label_keys,
                label_embeddings,
                threshold=threshold,
            )
            metadata["quantum_problem"] = problem_type
            classified_count += 1
            
            # Print progress every 10 images
            if classified_count % 10 == 0:
                print(f"  Classified {classified_count}/{total_images} images...")
        
        except Exception as e:
            print(f"[WARN] Failed to classify {img_name}: {e}")
            metadata["quantum_problem"] = "Unspecified quantum circuit"
    
    return classified_count


def main():
    """Main execution function."""
    print("="*80)
    print("Image Quantum Problem Population Script")
    print("="*80)
    
    # 1. Load images.json
    print(f"\n[1/4] Loading {IMAGES_JSON_PATH}...")
    images_data = load_images_json()
    
    if not images_data:
        print("[ERROR] No image data to process. Exiting.")
        return 1
    
    print(f"[OK] Loaded {len(images_data)} images")
    
    # 2. Load SBERT model (isolated instance for this script)
    print(f"\n[2/4] Loading SBERT model ({SBERT_MODEL_NAME})...")
    try:
        model = SentenceTransformer(SBERT_MODEL_NAME)
        print("[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load SBERT model: {e}")
        return 1
    
    # 3. Prepare label embeddings
    print(f"\n[3/4] Preparing label embeddings ({len(QUANTUM_PROBLEM_LABELS)} labels)...")
    try:
        label_keys, label_embeddings = prepare_label_embeddings(model)
        print(f"[OK] Prepared embeddings for {len(label_keys)} labels")
    except Exception as e:
        print(f"[ERROR] Failed to prepare label embeddings: {e}")
        return 1
    
    # 4. Classify and populate quantum problems
    print(f"\n[4/4] Classifying quantum problems...")
    classified_count = populate_quantum_problems(
        images_data,
        model,
        label_keys,
        label_embeddings,
        threshold=DEFAULT_THRESHOLD,
    )
    
    print(f"\n[OK] Classified {classified_count}/{len(images_data)} images")
    
    # 5. Save updated images.json
    print(f"\nSaving updated {IMAGES_JSON_PATH}...")
    if save_images_json(images_data):
        print("[OK] Successfully saved updated images.json")
        print("\n" + "="*80)
        print("✓ Quantum problem population complete!")
        print("="*80)
        return 0
    else:
        print("[ERROR] Failed to save images.json")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

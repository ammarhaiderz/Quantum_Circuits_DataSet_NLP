"""Test script for image metadata generation."""

import json
from core.image_extract_store import (
    generate_image_metadata,
    emit_image_record,
    finalize_images_output,
    _parse_figure_number_from_caption,
    _build_description_list,
)

def test_figure_number_parsing():
    """Test figure number extraction from captions."""
    print("\n=== Testing Figure Number Parsing ===")
    
    test_cases = [
        ("Fig. 5: Quantum circuit for error correction", 5),
        ("Figure 12: NCV quantum gate library", 12),
        ("Circuit diagram showing quantum gates", None),
        ("fig. 3 shows the quantum algorithm", 3),
        ("FIGURE 7: Error correction code", 7),
    ]
    
    for caption, expected in test_cases:
        result = _parse_figure_number_from_caption(caption)
        status = "✓" if result == expected else "✗"
        print(f"{status} Caption: '{caption[:50]}...' -> {result} (expected {expected})")

def test_metadata_generation():
    """Test complete metadata generation."""
    print("\n=== Testing Metadata Generation ===")
    
    test_cases = [
        {
            "arxiv_id": "2301.01234",
            "caption": "Fig. 5: Quantum circuit for error correction using Shor's 9-qubit code.",
            "preprocessed": "quantum circuit error correction shor 9 qubit code"
        },
        {
            "arxiv_id": "2302.05678",
            "caption": "Figure 3: Variational quantum eigensolver circuit.",
            "preprocessed": "variational quantum eigensolver circuit"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        metadata = generate_image_metadata(
            test["arxiv_id"],
            test["caption"],
            test["preprocessed"]
        )
        print(f"  arxiv_id: {metadata['arxiv_id']}")
        print(f"  figure_number: {metadata['figure_number']}")
        print(f"  page: {metadata['page']}")
        print(f"  quantum_problem: {metadata['quantum_problem']}")
        print(f"  description: {metadata['description']}")

def test_description_building():
    """Test description list building."""
    print("\n=== Testing Description Building ===")
    
    caption = "Fig. 2: Quantum circuit implementing Grover's algorithm."
    descriptions = _build_description_list(caption, latex_source=None)
    
    print(f"Caption: {caption}")
    print(f"Generated descriptions ({len(descriptions)}):")
    for i, desc in enumerate(descriptions, 1):
        print(f"  {i}. {desc[:80]}...")

def test_emit_and_finalize():
    """Test record emission and JSON finalization."""
    print("\n=== Testing Emit and Finalize ===")
    
    # Generate and emit test records
    test_records = [
        {
            "arxiv_id": "2301.01234",
            "caption": "Fig. 1: Quantum error correction circuit.",
            "preprocessed": "quantum error correction circuit"
        },
        {
            "arxiv_id": "2301.01234",
            "caption": "Fig. 2: Fault-tolerant quantum gates.",
            "preprocessed": "fault tolerant quantum gates"
        },
        {
            "arxiv_id": "2302.05678",
            "caption": "Figure 1: VQE circuit for molecular simulation.",
            "preprocessed": "vqe circuit molecular simulation"
        },
    ]
    
    print("\nEmitting test records to images.jsonl...")
    for record in test_records:
        metadata = generate_image_metadata(
            record["arxiv_id"],
            record["caption"],
            record["preprocessed"]
        )
        emit_image_record(metadata)
        print(f"  ✓ Emitted: {record['arxiv_id']} Fig. {metadata['figure_number']}")
    
    print("\nFinalizing images.json...")
    finalize_images_output()
    
    # Read and display result
    try:
        with open("data/images.json", "r", encoding="utf-8") as f:
            images_json = json.load(f)
        
        print(f"\n✓ Generated images.json with {len(images_json)} papers:")
        for arxiv_id, figures in images_json.items():
            print(f"  {arxiv_id}: {len(figures)} figures")
            for fig_key, fig_data in figures.items():
                print(f"    - {fig_key}: {fig_data.get('description', [''])[0][:60]}...")
    except Exception as e:
        print(f"✗ Failed to read images.json: {e}")

if __name__ == "__main__":
    print("="*80)
    print("Image Metadata Generation Test Suite")
    print("="*80)
    
    test_figure_number_parsing()
    test_metadata_generation()
    test_description_building()
    test_emit_and_finalize()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)

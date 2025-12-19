"""Check that PNGs in rendered_pdflatex/png are present as keys in data/circuits.json.

Usage:
    python tools/check_pngs_in_json.py

Outputs counts and a list of missing PNG filenames (if any).
"""
import json
from pathlib import Path
import sys


def main():
    repo_root = Path(__file__).resolve().parents[1]
    json_path = repo_root / 'data' / 'circuits.json'
    png_dir = repo_root / 'circuit_images' / 'rendered_pdflatex' / 'png'

    if not json_path.exists():
        print(f"ERROR: {json_path} not found")
        return 2
    if not png_dir.exists():
        print(f"ERROR: {png_dir} not found")
        return 2

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"ERROR: failed to parse {json_path}: {e}")
            return 2

    keys = set(data.keys()) if isinstance(data, dict) else set()
    pngs = sorted(p.name for p in png_dir.glob('*.png'))

    missing = [p for p in pngs if p not in keys]

    print(f"PNGs in folder: {len(pngs)}")
    print(f"Keys in data/circuits.json: {len(keys)}")
    print(f"Missing in JSON: {len(missing)}")

    if missing:
        print('\nFirst 200 missing PNGs:')
        for m in missing[:200]:
            print(m)
        return 1

    print('All PNGs are present as keys in data/circuits.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())

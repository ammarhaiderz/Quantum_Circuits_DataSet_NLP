from pathlib import Path
import json

JSONL_PATH = Path('data') / 'circuits.jsonl'
common_png_dir = Path('circuit_images') / 'rendered_pdflatex' / 'png'

with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        aid = rec.get('arxiv_id')
        pg = rec.get('page')
        print('record arxiv_id,page:', aid, pg)
        pattern = f"{aid}*p{pg}_*.png"
        print('glob pattern:', pattern)
        matches = list(common_png_dir.glob(pattern))
        print('dir exists:', common_png_dir.exists())
        print('matches count:', len(matches))
        for m in matches[:10]:
            print('  ', m.name)
        break

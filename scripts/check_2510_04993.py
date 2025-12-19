import json
from pathlib import Path
from core.circuit_store import find_caption_page_in_pdf

p = Path('data/circuits.jsonl')
if not p.exists():
    print('data/circuits.jsonl not found')
    raise SystemExit(1)

any_found = False
for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), start=1):
    if not line.strip():
        continue
    try:
        rec = json.loads(line)
    except Exception as e:
        print('line', i, 'json error', e)
        continue
    if rec.get('arxiv_id') == '2510.04993' and rec.get('page') is None:
        any_found = True
        caption = rec.get('descriptions')[0] if rec.get('descriptions') else ''
        print('BLOCK_ID:', rec.get('block_id'))
        print('RAW_BLOCK_FILE:', rec.get('raw_block_file'))
        print('CAPTION:', caption)
        try:
            res = find_caption_page_in_pdf('2510.04993', caption)
        except Exception as e:
            res = f'ERROR: {e}'
        print('DETECTED:', res)
        print('---')

if not any_found:
    print('No records with arxiv_id 2510.04993 and null page found in data/circuits.jsonl')

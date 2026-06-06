from pathlib import Path
import json

JSON_PATH = Path('data/circuits.json')
PNG_DIR = Path('circuit_images/rendered_pdflatex/png')
MAX = 250

if not JSON_PATH.exists():
    print('data/circuits.json not found')
    raise SystemExit(1)

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

if not isinstance(data, dict):
    print('data/circuits.json is not a dict')
    raise SystemExit(1)

n = len(data)
print(f'Current JSON entries: {n}')
if n < MAX:
    print('Fewer than MAX entries; nothing to remove')
    raise SystemExit(0)

keys = list(data.keys())
keep = set(keys[:MAX])
remove_keys = keys[MAX:]
print(f'JSON keys that would be removed: {len(remove_keys)}')
for k in remove_keys[:20]:
    print('  -', k)
if len(remove_keys) > 20:
    print('  ...')

# determine PNG files in folder not in keep
if not PNG_DIR.exists():
    print('PNG dir not found:', PNG_DIR)
    raise SystemExit(1)

pngs = [p.name for p in PNG_DIR.iterdir() if p.is_file()]
stray = [p for p in pngs if p not in keep]
print(f'PNG files in folder: {len(pngs)}; PNGs that would be removed: {len(stray)}')
for p in stray[:20]:
    print('  -', p)
if len(stray) > 20:
    print('  ...')

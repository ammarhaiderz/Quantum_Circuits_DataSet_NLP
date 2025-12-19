from pathlib import Path
import json, os

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
print(f'JSON keys to remove: {len(remove_keys)}')
removed = []
for k in remove_keys:
    p = PNG_DIR / k
    if p.exists() and p.is_file():
        try:
            os.remove(p)
            removed.append(p.name)
        except Exception as e:
            print('Failed to remove', p, e)

# also remove any stray PNGs not in keep
for p in PNG_DIR.iterdir():
    if not p.is_file():
        continue
    if p.name not in keep:
        if p.name not in removed:
            try:
                os.remove(p)
                removed.append(p.name)
            except Exception as e:
                print('Failed to remove', p, e)

# write trimmed JSON
new_data = {k: data[k] for k in keys if k in keep}
tmp = JSON_PATH.with_suffix('.json.tmp')
with open(tmp, 'w', encoding='utf-8') as wf:
    json.dump(new_data, wf, ensure_ascii=False, indent=2)
try:
    tmp.replace(JSON_PATH)
except Exception:
    try:
        tmp.rename(JSON_PATH)
    except Exception as e:
        print('Failed to replace JSON:', e)

print('Removed files:')
for r in removed:
    print(' -', r)
print('Done')

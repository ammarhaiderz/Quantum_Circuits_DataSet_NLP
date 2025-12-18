"""Scan raw .tex blocks and summarize gate tokens and LaTeX macro commands.
Writes outputs/gates_summary.json and prints a short summary.
"""
from pathlib import Path
import re
import json

ROOT = Path('circuit_images/live_blocks')
OUT = Path('outputs')
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / 'gates_summary.json'

tex_files = list(ROOT.rglob('raw_blocks/*.tex'))
if not tex_files:
    print('No raw .tex files found under', ROOT)
    raise SystemExit(0)

GATE_CONTENT_RE = re.compile(r"\\gate\{([^}]*)\}")
MACRO_RE = re.compile(r"\\([A-Za-z@]+)\b")

gate_counts = {}
gate_examples = {}
macro_counts = {}
macro_examples = {}

for f in tex_files:
    try:
        txt = f.read_text(encoding='utf-8')
    except Exception:
        continue

    # extract gate{...} contents
    for m in GATE_CONTENT_RE.finditer(txt):
        content = m.group(1)
        parts = re.split(r"[^A-Za-z0-9_\\]+", content)
        for p in parts:
            if not p:
                continue
            token = p.lstrip('\\').upper()
            gate_counts[token] = gate_counts.get(token, 0) + 1
            if token not in gate_examples:
                gate_examples[token] = []
            if len(gate_examples[token]) < 5:
                gate_examples[token].append(str(f))

    # extract macro commands used in the block
    for m in MACRO_RE.finditer(txt):
        name = m.group(1)
        # skip common environment names and begin/end
        if name.lower() in ('begin','end','document','quantikz','qcircuit'):
            continue
        macro_counts[name] = macro_counts.get(name, 0) + 1
        if name not in macro_examples:
            macro_examples[name] = []
        if len(macro_examples[name]) < 5:
            macro_examples[name].append(str(f))

report = {
    'total_tex_files': len(tex_files),
    'gate_counts': sorted([(k, v, gate_examples.get(k, [])) for k, v in gate_counts.items()], key=lambda x: x[1], reverse=True),
    'macro_counts': sorted([(k, v, macro_examples.get(k, [])) for k, v in macro_counts.items()], key=lambda x: x[1], reverse=True)
}

REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print('Wrote report to', REPORT)
print('Top gates:')
for k, v, ex in report['gate_counts'][:15]:
    print(f" - {k}: {v} (examples: {len(ex)})")
print('\nTop macros:')
for k, v, ex in report['macro_counts'][:20]:
    print(f" - {k}: {v} (examples: {len(ex)})")

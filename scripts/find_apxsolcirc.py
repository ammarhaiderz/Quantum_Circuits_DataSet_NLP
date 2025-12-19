import tarfile
import re

TAR='arxiv_cache/2404.14865.tar.gz'
TARGET='ApxSolCirc'
inc_re = re.compile(r"\\(?:input|include|subfile|import|subimport)\{([^}]+)\}")

with tarfile.open(TAR,'r:gz') as t:
    texs = [m for m in t.getmembers() if m.name.endswith('.tex')]
    found_direct = False
    for m in texs:
        f = t.extractfile(m)
        if f is None:
            continue
        txt = f.read().decode('utf-8','ignore')
        if TARGET in txt:
            found_direct = True
            print(f"FOUND literal '{TARGET}' in: {m.name}")
            for i,l in enumerate(txt.splitlines(),1):
                if TARGET in l:
                    print(f"  {m.name}:{i}: {l}")
    if not found_direct:
        print(f"No literal '{TARGET}' found in any .tex. Searching include directives...")
        for m in texs:
            f = t.extractfile(m)
            if f is None:
                continue
            txt = f.read().decode('utf-8','ignore')
            for i,l in enumerate(txt.splitlines(),1):
                for mo in inc_re.finditer(l):
                    tgt = mo.group(1).strip()
                    # normalize: drop ./ and .tex
                    norm = tgt
                    if norm.startswith('./'):
                        norm = norm[2:]
                    if norm.endswith('.tex'):
                        norm = norm[:-4]
                    if norm.endswith('/' + TARGET) or norm == TARGET or norm.endswith(TARGET):
                        print(f"{m.name}:{i}: include -> {tgt}  (norm -> {norm})")

print('\nNow performing include traversal from chosen main.tex')
with tarfile.open(TAR,'r:gz') as t:
    texs = {m.name: t.extractfile(m).read().decode('utf-8','ignore') for m in t.getmembers() if m.name.endswith('.tex')}

main_candidates = list(texs.keys())
chosen = None
for name,txt in texs.items():
    if '\\documentclass' in txt:
        chosen = name
        break
if chosen is None:
    for name,txt in texs.items():
        if '\\begin{document}' in txt:
            chosen = name
            break
if chosen is None:
    if 'main.tex' in texs:
        chosen = 'main.tex'
    else:
        chosen = main_candidates[0]
print('Chosen main:', chosen)

included = set()
queue = [chosen]
while queue:
    cur = queue.pop(0)
    if cur in included:
        continue
    included.add(cur)
    txt = texs[cur]
    for mo in inc_re.finditer(txt):
        tgt = mo.group(1).strip()
        norm = tgt
        if norm.startswith('./'):
            norm = norm[2:]
        if norm.endswith('.tex'):
            norm = norm[:-4]
        for candidate in texs.keys():
            cand_norm = candidate
            if cand_norm.endswith('.tex'):
                cand_norm2 = cand_norm[:-4]
            else:
                cand_norm2 = cand_norm
            if cand_norm2 == norm or cand_norm2.endswith('/'+norm) or cand_norm2.endswith(norm):
                queue.append(candidate)

print('\nIncluded files from traversal:')
for x in sorted(included):
    print(' ', x)

matches = [x for x in included if x.endswith('ApxSolCirc.tex') or x.endswith('/ApxSolCirc.tex') or x.endswith('ApxSolCirc')]
print('\nApxSolCirc in included traversal?:', bool(matches))
if matches:
    print('Matched files:', matches)
else:
    print('No match in traversal')

# Build include graph (mapping from file -> list of candidate filenames it includes)
include_map = {name: [] for name in texs.keys()}
for name, txt in texs.items():
    for mo in inc_re.finditer(txt):
        tgt = mo.group(1).strip()
        norm = tgt
        if norm.startswith('./'):
            norm = norm[2:]
        if norm.endswith('.tex'):
            norm = norm[:-4]
        for candidate in texs.keys():
            cand_norm = candidate[:-4] if candidate.endswith('.tex') else candidate
            if cand_norm == norm or cand_norm.endswith('/'+norm) or cand_norm.endswith(norm):
                include_map[name].append(candidate)

# Find path(s) from chosen main to any ApxSolCirc match
targets = set(matches)
from collections import deque
q = deque([[chosen]])
visited = set([chosen])
paths = []
while q:
    path = q.popleft()
    node = path[-1]
    if node in targets:
        paths.append(path)
        continue
    for nb in include_map.get(node,[]):
        if nb in path:
            continue
        q.append(path + [nb])

print('\nInclude path(s) from main to ApxSolCirc:')
if paths:
    for p in paths:
        for i,step in enumerate(p,1):
            print(f"  {i}. {step}")
        print('')
else:
    print('  No path found')

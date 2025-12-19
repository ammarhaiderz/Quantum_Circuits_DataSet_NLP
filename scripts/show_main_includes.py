import sys
import tarfile

def show_includes(paper_id: str):
    tar_path = f"arxiv_cache/{paper_id}.tar.gz"
    try:
        t = tarfile.open(tar_path, mode='r:gz')
    except Exception as e:
        print(f"Failed to open {tar_path}: {e}")
        return 2

    # prefer member named main.tex, otherwise try files with documentclass or begin{document}
    members = [m.name for m in t.getmembers() if m.name.endswith('.tex')]
    if not members:
        print("No .tex members in tarball")
        return 1

    def read_member(name):
        try:
            return t.extractfile(name).read().decode('utf-8', 'ignore')
        except Exception:
            return ''

    # find main.tex if present
    if 'main.tex' in members:
        main_name = 'main.tex'
    else:
        # pick a file that contains \documentclass, else one with \begin{document}
        main_name = None
        for name in members:
            txt = read_member(name)
            if '\\documentclass' in txt:
                main_name = name
                break
        if not main_name:
            for name in members:
                txt = read_member(name)
                if '\\begin{document}' in txt:
                    main_name = name
                    break
    if not main_name:
        main_name = members[0]

    main_txt = read_member(main_name)
    print(f"Main file chosen: {main_name}\n")

    # print lines that include or reference ApxSolCirc.tex or general include commands
    lines = main_txt.splitlines()
    for i, line in enumerate(lines, start=1):
        if 'ApxSolCirc' in line or '\\input' in line or '\\include' in line or '\\subfile' in line or '\\import' in line:
            print(f"{i}: {line}")

    # Additionally show normalized references matching ApxSolCirc in any include statements
    print('\nChecking include targets that may reference ApxSolCirc:')
    import re
    inc_re = re.compile(r"\\(?:input|include|subfile|import|subimport)\{([^}]+)\}")
    for i, line in enumerate(lines, start=1):
        for m in inc_re.finditer(line):
            target = m.group(1).strip()
            if 'ApxSolCirc' in target or target.endswith('ApxSolCirc') or target.endswith('ApxSolCirc.tex'):
                print(f"{i}: include -> {target}")

    return 0

if __name__ == '__main__':
    pid = sys.argv[1] if len(sys.argv) > 1 else '2404.14865'
    sys.exit(show_includes(pid))

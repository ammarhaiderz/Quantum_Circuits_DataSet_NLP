from pathlib import Path
import json
from typing import List


def cleanup_extra_pngs(json_path: str = 'data/circuits.json', png_dir: str = 'circuit_images/rendered_pdflatex/png', max_images: int = 250) -> List[str]:
    """Trim canonical JSON to first `max_images` keys and remove PNGs not kept.

    Returns list of removed filenames.
    """
    jp = Path(json_path)
    pd = Path(png_dir)
    removed: List[str] = []

    if not jp.exists() or not pd.exists():
        return removed

    try:
        with jp.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return removed

    if not isinstance(data, dict):
        return removed

    if len(data) < max_images:
        return removed

    keys = list(data.keys())
    keep = set(keys[:max_images])

    # remove PNG files not in the kept set
    try:
        for p in pd.iterdir():
            if not p.is_file():
                continue
            if p.name not in keep:
                try:
                    p.unlink()
                    removed.append(p.name)
                except Exception:
                    pass
    except Exception:
        pass

    # write trimmed JSON atomically
    new_data = {k: data[k] for k in keys if k in keep}
    tmp = jp.with_suffix('.json.tmp')
    try:
        with tmp.open('w', encoding='utf-8') as wf:
            json.dump(new_data, wf, ensure_ascii=False, indent=2)
        try:
            tmp.replace(jp)
        except Exception:
            try:
                tmp.rename(jp)
            except Exception:
                pass
    except Exception:
        pass

    return removed

"""Generate counts of extracted circuits per paper from circuits.jsonl.

For each ID in paper_list_36.txt:
- Count how many records in data/circuits.jsonl share the same arxiv_id.
- Write that count to latex_render_image_count; if absent, write 0.

Output: paper_list_counts_36.csv with columns arxiv_id, latex_render_image_count.
Non-quant-ph (from arxiv_category_cache.json) rows are left blank in the count column.
Papers beyond the checkpoint index in latex_code_circuit_checkpoint.json are left blank.
"""

from pathlib import Path
import json
import tarfile
import io

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "paper_list_36.txt"
OUTPUT = PROJECT_ROOT / "paper_list_counts_36.csv"
CIRCUITS = PROJECT_ROOT / "data" / "circuits.jsonl"
CATEGORY_CACHE = PROJECT_ROOT / "data" / "arxiv_category_cache.json"
CHECKPOINT = PROJECT_ROOT / "data" / "latex_code_circuit_checkpoint.json"
ARXIV_CACHE = PROJECT_ROOT / "arxiv_cache"


def normalize(arxiv_id: str) -> str:
	arxiv_id = arxiv_id.strip()
	if arxiv_id.startswith("arXiv:"):
		arxiv_id = arxiv_id.split(":", 1)[1]
	base, sep, suffix = arxiv_id.partition("v")
	if sep and suffix.isdigit():
		return base
	return arxiv_id


def load_paper_ids() -> list[str]:
	return [line.strip() for line in SOURCE.read_text().splitlines() if line.strip()]


def load_category_flags() -> dict[str, bool]:
	if not CATEGORY_CACHE.exists():
		return {}
	try:
		return json.loads(CATEGORY_CACHE.read_text())
	except json.JSONDecodeError:
		return {}


def load_checkpoint_limit() -> int | None:
	if not CHECKPOINT.exists():
		return None
	try:
		data = json.loads(CHECKPOINT.read_text())
		return int(data.get("total_papers_processed"))
	except (json.JSONDecodeError, ValueError, TypeError):
		return None


def count_circuits() -> dict[str, int]:
	counts: dict[str, int] = {}
	if not CIRCUITS.exists():
		return counts
	with CIRCUITS.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue
			aid = obj.get("arxiv_id")
			if not aid:
				continue
			aid_norm = normalize(str(aid))
			counts[aid_norm] = counts.get(aid_norm, 0) + 1
	return counts


def validate_tar_files(df: pd.DataFrame) -> None:
	"""Mark papers with corrupted tar files by blanking their count column."""
	corrupted = []
	for idx, row in df.iterrows():
		arxiv_id = row["arxiv_id"]
		count_val = row["latex_render_image_count"]
		
		# Skip if already blank (non-quantum or beyond checkpoint)
		if count_val == "" or pd.isna(count_val):
			continue
		
		# Try to open tar file (stored directly in arxiv_cache/)
		norm_id = normalize(arxiv_id)
		tar_path = ARXIV_CACHE / f"{norm_id}.tar.gz"
		
		if not tar_path.exists():
			continue  # No tar file yet, keep as-is
		
		try:
			# Open directly like extraction pipeline does
			with open(tar_path, "rb") as f:
				tar = tarfile.open(fileobj=f, mode="r:gz")
				tar.close()
		except Exception:
			# Any exception means corrupted/invalid - blank the count
			df.at[idx, "latex_render_image_count"] = ""
			corrupted.append(arxiv_id)
	
	print(f"[TAR VALIDATION] Found {len(corrupted)} corrupted tar files till checkpoint")
	if corrupted:
		print(f"  Corrupted papers: {corrupted[:10]}{'...' if len(corrupted) > 10 else ''}")


def main() -> None:
	paper_ids = load_paper_ids()
	normalized_ids = [normalize(pid) for pid in paper_ids]
	circuit_counts = count_circuits()
	category_flags = load_category_flags()
	checkpoint_limit = load_checkpoint_limit()

	rows = []
	for idx, (raw, norm) in enumerate(zip(paper_ids, normalized_ids)):
		# Apply checkpoint (total_papers_processed is 1-based). Blank anything beyond that count.
		if checkpoint_limit is not None and (idx + 1) > checkpoint_limit:
			rows.append({"arxiv_id": raw, "latex_render_image_count": ""})
			continue

		is_quant = category_flags.get(norm, True)
		if not is_quant:
			count = ""
		else:
			count = circuit_counts.get(norm, 0)
		rows.append({"arxiv_id": raw, "latex_render_image_count": count})

	df = pd.DataFrame(rows, columns=["arxiv_id", "latex_render_image_count"])
	
	# Validate tar files and mark corrupted ones
	validate_tar_files(df)
	
	df.to_csv(OUTPUT, index=False)
	print(
		f"Wrote {OUTPUT} with {len(df)} rows; circuits file exists: {CIRCUITS.exists()}, "
		f"category cache exists: {CATEGORY_CACHE.exists()}, checkpoint: {checkpoint_limit}"
	)


if __name__ == "__main__":
	main()
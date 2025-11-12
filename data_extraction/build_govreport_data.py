import json
import os
import random
import re
from datasets import load_dataset

TOTAL_DOCS = 300
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
MIN_PARAGRAPHS = 3
SEED = 20250112  # freeze your sample

OUT_DIR = "data/govreport_subset"
SPLIT_DIR = os.path.join(OUT_DIR, "splits")


def normalize_text(text: str) -> str:
    """Standardize newlines but preserve paragraph boundaries (double newlines)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ newlines to exactly 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim leading/trailing whitespace but keep internal paragraph structure
    return text.strip()


def sanitize_filename(doc_id: str) -> str:
    """Keep only safe characters for filenames."""
    # Replace any non-alphanumeric (except _ and -) with underscore
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", str(doc_id))


def is_usable(report: str) -> bool:
    """Check if report has enough paragraphs and length."""
    if not report:
        return False
    
    # Count paragraphs BEFORE normalization (on original text)
    # Paragraphs are separated by double newlines
    paragraph_count = len([p for p in report.split("\n\n") if p.strip()])
    
    # Also check word count
    word_count = len(report.split())
    
    return paragraph_count >= MIN_PARAGRAPHS and word_count >= 120


def main():
    random.seed(SEED)

    dataset = load_dataset("ccdv/govreport-summarization")
    train_pool = dataset["train"]
    indices = list(range(len(train_pool)))
    random.shuffle(indices)

    selected = []
    for idx in indices:
        example = train_pool[idx]
        report = example["report"]
        
        # Check usability BEFORE normalization
        if is_usable(report):
            normalized = normalize_text(report)
            doc_id = example.get("report_id", idx)
            selected.append((doc_id, normalized))
        if len(selected) >= TOTAL_DOCS:
            break

    if len(selected) < TOTAL_DOCS:
        raise RuntimeError(
            f"Only found {len(selected)} usable docs (wanted {TOTAL_DOCS}). "
            "Loosen filters or increase pool."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    split_sizes = {
        split: int(TOTAL_DOCS * ratio)
        for split, ratio in SPLIT_RATIOS.items()
    }
    assigned = sum(split_sizes.values())
    if assigned != TOTAL_DOCS:
        split_sizes["train"] += TOTAL_DOCS - assigned

    cursor = 0
    split_doc_ids = {}
    for split, size in split_sizes.items():
        doc_ids = []
        for i in range(size):
            doc_id, report = selected[cursor + i]
            safe_id = sanitize_filename(doc_id)
            out_path = os.path.join(OUT_DIR, f"{split}_{safe_id}.txt")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(report)
            doc_ids.append(str(doc_id))  # Store original ID in manifest
        split_doc_ids[split] = doc_ids
        cursor += size

    for split, doc_ids in split_doc_ids.items():
        split_file = os.path.join(SPLIT_DIR, f"{split}.json")
        with open(split_file, "w", encoding="utf-8") as fh:
            json.dump(doc_ids, fh, indent=2)

    print(
        "Saved documents:",
        ", ".join(f"{split}={len(doc_ids)}" for split, doc_ids in split_doc_ids.items()),
    )
    print(f"Outputs in '{OUT_DIR}' and split lists in '{SPLIT_DIR}'.")


if __name__ == "__main__":
    main()
# FactShield (CPU) — Evidence-Grounded Summarization

**Goal**: RAG + must-cite summaries of long public documents with sentence-level factuality verification on CPU-only hardware.

## Quick Start
```bash
conda env create -f environment.yml
conda activate factshield
python -m spacy download en_core_web_sm
make setup
````

## Data
### Generate GovReport subset (run once)
```bash
python data_extraction/build_govreport_data.py
```
This downloads a seeded sample of `ccdv/govreport-summarization` into `data/govreport_subset/` and writes deterministic train/dev/test manifests.


## Pipeline

```bash
make chunk         # chunk raw docs → cache/chunks
make embed         # sentence-embeddings → cache/emb
make index         # FAISS CPU index → cache/faiss
make retrieve      # top-k chunks per doc → cache/retrieval
make baseline_textrank
make baseline_distilbart
make improved_rag  # RAG + must-cite summaries → runs/improved
make verify_nli    # Sentence-level NLI support → runs/verify
make eval_all      # ROUGE + bootstrap CI table → eval/
make fact_eval     # Factuality summary with CIs → eval/
make app           # launch Gradio demo
```

Or reproduce end-to-end:

```bash
make run_all
```

## Human Evaluation Template

Create `eval/annotations.csv` with columns:
`doc_id,sent_id,text,label_annotator1,label_annotator2,error_type`

* `label_*` in `{SUPPORTED,LOW-CONF}`
* `error_type` (FRANK-style): `entity, numeric, predicate, discourse, unverifiable, other`

Then run:

```bash
make human_eval
```

## Reproducibility

We expose a global seed in `src/seed.py`:

```python
from src.seed import set_seed
set_seed(42)
```

Call it at the top of any new scripts you add.

## Outputs

* `runs/baselines/*.jsonl` – baseline summaries
* `runs/improved/rag_cited.jsonl` – improved summaries with `[CIT:*]`
* `runs/verify/nli_scores.jsonl` – per-sentence NLI status + support ratio
* `eval/rouge_*.csv`, `eval/summary_table.csv` – ROUGE with 95% CIs
* `eval/factuality_summary.csv` – factuality mean with 95% CIs
* `eval/human_eval_summary.json` – Cohen’s κ and error tallies (if provided)


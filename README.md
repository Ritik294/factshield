# FactShield — Evidence-Grounded Summarization

**Course**: Data641 (Fall 2025)

**Contributors**: Swattik Maiti, Ritik Pratap Singh

**Goal**: RAG + must-cite summaries of long public documents with sentence-level factuality verification on CPU-only hardware.

## Quick Start
```bash
conda env create -f environment.yml
conda activate factshield
python -m spacy download en_core_web_sm
make setup
````

> **Note:** If `make` isn't available on your OS, install it (e.g., via Chocolatey or Scoop on Windows) or run the equivalent Python commands shown in the Makefile.

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

## Extended Evaluation Pipeline

After running the basic pipeline, you can run additional evaluation and analysis:

# 1. Run new baseline (Lead-3 extractive)
```bash
make baseline_lead3
```

# 2. Evaluate Lead-3 baseline
```bash
python -m src.eval_rouge --refs data/govreport_subset --hyps runs/baselines/lead3.jsonl --out eval/rouge_lead3.csv
```

# 3. Enhanced statistics (with statistical tests and per-document breakdown)
```bash
make eval_stats_enhanced
```

# 4. Error analysis (categorize NLI failures)
```bash
make error_analysis
```

# 5. Qualitative examples (best/worst summaries)
```bash
make qualitative
```

# 6. Citation quality metrics
```bash
make citation_quality
```

# 7. Bias analysis (gender/demographic terms)
```bash
make eval_bias
```

# 8. Limitations analysis (failure cases, citation issues)
```bash
make eval_limitations
```

# 9. BERTScore evaluation (semantic similarity - takes ~7 minutes)
```bash
make eval_bertscore
```
# 10. Ablation studies (run separately, each takes time)
```bash
make ablation_threshold  # Fastest (~1 second)
make ablation_topk       # Medium (~20-25 minutes)
make ablation_chunksize  # Slower (~10-15 minutes)
```

Instead of running steps 1-8 individually, you can use the comprehensive evaluation target:
```bash
make eval_comprehensive
```

This automatically runs:
- `baseline_lead3` (if not already run)
- Lead-3 ROUGE evaluation
- Enhanced statistics with statistical tests
- Error analysis
- Qualitative examples
- Citation quality
- Bias analysis
- Limitations analysis

**Note**: This does NOT include BERTScore or ablation studies (steps 9-10) as they are time-consuming. Run those separately if needed.

## Tests

Run `pytest` from the repository root to execute the unit/integration checks in `tests/`.
```bash
pytest
```

## Human Evaluation

**Manual Annotation Process**: The `eval/annotations.csv` file was manually created by:
1. Sampling summaries from `runs/improved/rag_cited.jsonl`
2. Extracting sentences from selected summaries
3. Having two annotators independently label each sentence as `SUPPORTED` or `LOW-CONF`
4. Recording error types for sentences marked as `LOW-CONF`

**CSV Format**: `eval/annotations.csv` contains columns:
- `doc_id` – document identifier
- `sent_id` – sentence number within summary
- `text` – sentence text
- `label_annotator1` – label from annotator 1: `SUPPORTED` or `LOW-CONF`
- `label_annotator2` – label from annotator 2: `SUPPORTED` or `LOW-CONF`
- `error_type` – error category (if `LOW-CONF`): `entity`, `numeric`, `predicate`, `discourse`, `unverifiable`, `other`

**Running Human Evaluation**: This computes inter-annotator agreement (Cohen's κ) and error type statistics from the manual annotations.

Then run:

```bash
make human_eval
```

## Reproducibility

We ensure reproducibility through:

1. **Fixed Random Seed**: Global seed set to 42 in `src/seed.py`:
   ```python
   from src.seed import set_seed
   set_seed(42)
   ```
2. **Deterministic Data Splits**: The data extraction script uses a fixed seed to generate reproducible train/dev/test splits saved in `data/govreport_subset/`.

3. **Model Versions**: All models are pinned to specific versions:
   - DistilBART: `sshleifer/distilbart-cnn-12-6`
   - RoBERTa-large-MNLI: `roberta-large-mnli`
   - Sentence-transformers: `all-MiniLM-L6-v2`

4. **Environment**: Use `environment.yml` or `requirements.txt` to ensure consistent package versions.

To reproduce exact results:
1. Use the same Python version (3.8+)
2. Install dependencies from `requirements.txt`
3. Run data extraction first (generates deterministic splits)
4. Run pipeline steps in order



## Outputs

* `runs/baselines/*.jsonl` – baseline summaries (textrank, distilbart, lead3)
* `runs/improved/rag_cited.jsonl` – improved summaries with `[CIT:*]`
* `runs/verify/nli_scores.jsonl` – per-sentence NLI status + support ratio
* `runs/ablation/*.csv` – ablation study results (top-k, chunk size, threshold)
* `eval/rouge_*.csv`, `eval/summary_table.csv` – ROUGE with 95% CIs
* `eval/summary_table_statistical_tests.csv` – statistical significance tests
* `eval/summary_table_per_doc.csv` – per-document ROUGE breakdown
* `eval/factuality_summary.csv` – factuality mean with 95% CIs
* `eval/error_analysis.csv` – error type distribution
* `eval/error_analysis_detailed.csv` – detailed error examples
* `eval/bias_analysis.csv` – gender/demographic term frequency
* `eval/citation_quality.csv` – citation coverage and accuracy metrics
* `eval/limitations_analysis.csv` – failure case analysis
* `eval/limitations_analysis_summary.json` – limitations summary statistics
* `eval/qualitative_examples.json` – best/worst summary examples
* `eval/bertscore_rag.csv` – BERTScore semantic similarity scores
* `eval/human_eval_summary.json` – Cohen's κ and error tallies (if provided)


## Evaluation Notes

**ROUGE Scores**: We use the first ~3 paragraphs of each document as pseudo-references. This is a limitation since these may not be ideal gold summaries. Lower ROUGE scores may reflect:
- Pseudo-reference quality (first paragraphs may not capture document essence)
- Abstractive vs extractive mismatch (our models generate abstractive summaries)
- Domain-specific language in government reports

**Factuality Verification**: Requires citations in summaries. If `support_ratio` is 0.0, check that RAG summaries contain `[CIT:doc:chunk]` tags.

**Statistical Tests**: Enhanced evaluation includes t-tests comparing RAG vs baselines. Results are saved in `eval/summary_table_statistical_tests.csv`.



## Interface Demo

Launch the Gradio UI:
```bash
make app
# or
python -m app.app
```
Once running, open `http://127.0.0.1:7860` (or `http://localhost:7860`) in your browser.  
Paste a long document, click “Summarize with citations + verify,” and you’ll get:
- A summary with `[CIT:doc:chunk]` markers (click them to see citations).
- An overall factuality support ratio.
- Sentence-level verification details (supported vs low-confidence).



## Troubleshooting

**Issue**: `DatasetNotFoundError` when running data extraction
- **Solution**: Ensure internet connection and Hugging Face access. Try: `huggingface-cli login`

**Issue**: `ModuleNotFoundError` when running scripts
- **Solution**: Ensure conda environment is activated: `conda activate factshield`

**Issue**: Low factuality scores (0% support ratio)
- **Solution**: Check that RAG summaries contain `[CIT:doc:chunk]` tags. Verify citations are being generated.

**Issue**: Out of memory errors
- **Solution**: Reduce batch sizes in scripts or use smaller models.

**Issue**: `make` command not found (Windows)
- **Solution**: Install via Chocolatey (`choco install make`) or run Python commands directly from Makefile.

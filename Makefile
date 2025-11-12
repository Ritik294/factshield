# Use tabs (not spaces) before each command!
PY=python

.PHONY: setup chunk embed index retrieve baseline_textrank baseline_distilbart \
        improved_rag verify_nli eval_all app human_eval fact_eval run_all

setup:
	@echo "Creating folders..."
	@mkdir -p cache/chunks cache/emb cache/faiss cache/retrieval runs/baselines runs/improved runs/verify eval
	@echo "Done."

chunk:
	$(PY) -m src.chunker --input_dir data/govreport_subset --out cache/chunks/chunks.jsonl

embed:
	$(PY) -m src.embed_retrieve build-emb --chunks cache/chunks/chunks.jsonl --emb_out cache/emb/emb.npy --meta_out cache/emb/meta.json

index:
	$(PY) -m src.embed_retrieve build-index --emb cache/emb/emb.npy --index_out cache/faiss/index.faiss

retrieve:
	$(PY) -m src.embed_retrieve retrieve --chunks cache/chunks/chunks.jsonl --meta cache/emb/meta.json --index cache/faiss/index.faiss --top_k 3 --out cache/retrieval/topk.jsonl

baseline_textrank:
	$(PY) -m src.summarizers textrank --input_dir data/govreport_subset --out runs/baselines/textrank.jsonl

baseline_distilbart:
	$(PY) -m src.summarizers distilbart --chunks cache/chunks/chunks.jsonl --out runs/baselines/distilbart.jsonl

improved_rag:
	$(PY) -m src.summarizers rag --chunks cache/chunks/chunks.jsonl --topk cache/retrieval/topk.jsonl --out runs/improved/rag_cited.jsonl

verify_nli:
	$(PY) -m src.verifier nli --summ runs/improved/rag_cited.jsonl --chunks cache/chunks/chunks.jsonl --out runs/verify/nli_scores.jsonl

eval_all:
	$(PY) -m src.eval_rouge --refs data/govreport_subset --hyps runs/baselines/textrank.jsonl --out eval/rouge_textrank.csv
	$(PY) -m src.eval_rouge --refs data/govreport_subset --hyps runs/baselines/distilbart.jsonl --out eval/rouge_distilbart.csv
	$(PY) -m src.eval_rouge --refs data/govreport_subset --hyps runs/improved/rag_cited.jsonl --out eval/rouge_rag.csv
	$(PY) -m src.eval_stats --in eval/ --out eval/summary_table.csv

app:
	$(PY) -m app.app


human_eval:
	@echo "Note: expects eval/annotations.csv with columns described in src/eval_human.py"
	$(PY) -m src.eval_human --csv eval/annotations.csv --out eval/human_eval_summary.json

fact_eval:
	$(PY) -m src.eval_factuality --nli runs/verify/nli_scores.jsonl --out eval/factuality_summary.csv

run_all:
	$(MAKE) setup && $(MAKE) chunk && $(MAKE) embed && $(MAKE) index && $(MAKE) retrieve \
	&& $(MAKE) baseline_textrank && $(MAKE) baseline_distilbart \
	&& $(MAKE) improved_rag && $(MAKE) verify_nli && $(MAKE) eval_all && $(MAKE) fact_eval

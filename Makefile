PY=python

.PHONY: setup chunk embed index retrieve baseline_textrank baseline_distilbart baseline_lead3 \
        improved_rag verify_nli eval_all app human_eval fact_eval run_all \
        ablation_topk ablation_chunksize ablation_threshold error_analysis qualitative \
        citation_quality eval_stats_enhanced eval_bertscore eval_bias eval_limitations \
        eval_comprehensive

setup:
	@echo "Creating folders..."
	@python -c "import os; os.makedirs('cache/chunks', exist_ok=True); os.makedirs('cache/emb', exist_ok=True); os.makedirs('cache/faiss', exist_ok=True); os.makedirs('cache/retrieval', exist_ok=True); os.makedirs('runs/baselines', exist_ok=True); os.makedirs('runs/improved', exist_ok=True); os.makedirs('runs/verify', exist_ok=True); os.makedirs('eval', exist_ok=True)"
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
	$(PY) -m src.eval_rouge --refs data/govreport_subset --hyps runs/baselines/lead3.jsonl --out eval/rouge_lead3.csv
	$(PY) -m src.eval_rouge --refs data/govreport_subset --hyps runs/improved/rag_cited.jsonl --out eval/rouge_rag.csv
	$(PY) -m src.eval_stats --in eval/ --out eval/summary_table.csv

app:
	$(PY) -m app.app

human_eval:
	@echo "Note: expects eval/annotations.csv with columns described in src/eval_human.py"
	$(PY) -m src.eval_human --csv eval/annotations.csv --out eval/human_eval_summary.json

fact_eval:
	$(PY) -m src.eval_factuality --nli runs/verify/nli_scores.jsonl --out eval/factuality_summary.csv

eval_comprehensive:
	$(MAKE) baseline_lead3
	$(PY) -m src.eval_rouge --refs data/govreport_subset --hyps runs/baselines/lead3.jsonl --out eval/rouge_lead3.csv
	$(MAKE) eval_stats_enhanced
	$(MAKE) error_analysis
	$(MAKE) qualitative
	$(MAKE) citation_quality
	$(MAKE) eval_bias
	$(MAKE) eval_limitations
	@echo "Comprehensive evaluation complete!"

baseline_lead3:
	$(PY) -m src.summarizers lead3 --input_dir data/govreport_subset --out runs/baselines/lead3.jsonl

ablation_topk:
	@python -c "import os; os.makedirs('runs/ablation', exist_ok=True)"
	$(PY) -m src.ablation_studies topk --chunks cache/chunks/chunks.jsonl --meta cache/emb/meta.json --index cache/faiss/index.faiss --out runs/ablation/topk_results.csv

ablation_chunksize:
	@python -c "import os; os.makedirs('runs/ablation', exist_ok=True)"
	$(PY) -m src.ablation_studies chunksize --input_dir data/govreport_subset --out runs/ablation/chunksize_results.csv

ablation_threshold:
	@python -c "import os; os.makedirs('runs/ablation', exist_ok=True)"
	$(PY) -m src.ablation_studies threshold --nli runs/verify/nli_scores.jsonl --out runs/ablation/threshold_results.csv

error_analysis:
	$(PY) -m src.eval_error_analysis --nli runs/verify/nli_scores.jsonl --chunks cache/chunks/chunks.jsonl --out eval/error_analysis.csv

qualitative:
	$(PY) -m src.eval_qualitative sample --rouge eval/rouge_rag.csv --summ runs/improved/rag_cited.jsonl --nli runs/verify/nli_scores.jsonl --out eval/qualitative_examples.json

citation_quality:
	$(PY) -m src.eval_qualitative citations --summ runs/improved/rag_cited.jsonl --nli runs/verify/nli_scores.jsonl --out eval/citation_quality.csv

eval_stats_enhanced:
	$(PY) -m src.eval_stats --in eval/ --out eval/summary_table.csv --per_doc

eval_bertscore:
	$(PY) -m src.eval_bertscore --refs data/govreport_subset --hyps runs/improved/rag_cited.jsonl --out eval/bertscore_rag.csv

eval_bias:
	$(PY) -m src.eval_bias --summ runs/improved/rag_cited.jsonl --out eval/bias_analysis.csv

eval_limitations:
	$(PY) -m src.eval_limitations --nli runs/verify/nli_scores.jsonl --summ runs/improved/rag_cited.jsonl --chunks cache/chunks/chunks.jsonl --out eval/limitations_analysis.csv

run_all:
	$(MAKE) setup && $(MAKE) chunk && $(MAKE) embed && $(MAKE) index && $(MAKE) retrieve \
	&& $(MAKE) baseline_textrank && $(MAKE) baseline_distilbart \
	&& $(MAKE) improved_rag && $(MAKE) verify_nli && $(MAKE) eval_all && $(MAKE) fact_eval
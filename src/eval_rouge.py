import argparse, os, json
from rouge_score import rouge_scorer
from src.utils import read_jsonl

# Expect references as raw docs in data/<split>/DOCID.txt (first 3 paragraphs as pseudo-ref if no gold)

def load_ref_text(ref_dir, doc_id):
    fp = os.path.join(ref_dir, doc_id + '.txt')
    with open(fp, 'r', encoding='utf-8') as f:
        txt = f.read()
    # crude pseudo-reference: first ~3 paragraphs
    paras = [p.strip() for p in txt.split('\n\n') if p.strip()]
    return " ".join(paras[:3])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refs', required=True)
    ap.add_argument('--hyps', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

    rows = []
    for row in read_jsonl(args.hyps):
        ref = load_ref_text(args.refs, row['doc_id'])
        hyp = row['summary']
        sc = scorer.score(ref, hyp)
        rows.append({'doc_id': row['doc_id'], **{k:v.fmeasure for k,v in sc.items()}})
    import pandas as pd
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
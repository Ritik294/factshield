import argparse, json, numpy as np, pandas as pd
from src.utils import read_jsonl

# Aggregate verifier outputs → per-model factuality with bootstrap CIs

def bootstrap_mean(x, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return float(np.mean(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nli', required=True)   # runs/verify/nli_scores.jsonl
    ap.add_argument('--out', required=True)   # eval/factuality_summary.csv
    args = ap.parse_args()

    support = []
    per_doc = []
    for row in read_jsonl(args.nli):
        support.append(row['support_ratio'])
        per_doc.append({'doc_id': row['doc_id'], 'support_ratio': row['support_ratio']})

    mean_, lo, hi = bootstrap_mean(np.array(support))
    df = pd.DataFrame(per_doc)
    df2 = pd.DataFrame([{'metric':'support_ratio','mean':mean_, 'ci_low':lo, 'ci_high':hi, 'n':len(support)}])
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out.replace('.csv','_per_doc.csv'), index=False)
    df2.to_csv(args.out, index=False)
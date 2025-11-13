"""
BERTScore evaluation: semantic similarity evaluation.
"""
import argparse
import os
import pandas as pd
from bert_score import score
from src.utils import read_jsonl

def load_ref_text(ref_dir, doc_id):
    """Load reference text (first 3 paragraphs)."""
    fp = os.path.join(ref_dir, doc_id + '.txt')
    with open(fp, 'r', encoding='utf-8') as f:
        txt = f.read()
    paras = [p.strip() for p in txt.split('\n\n') if p.strip()]
    return " ".join(paras[:3])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--refs', required=True, help='Reference directory')
    ap.add_argument('--hyps', required=True, help='Hypothesis JSONL path')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()
    
    # Load summaries
    refs = []
    hyps = []
    doc_ids = []
    
    for row in read_jsonl(args.hyps):
        doc_id = row['doc_id']
        ref = load_ref_text(args.refs, doc_id)
        hyp = row['summary']
        refs.append(ref)
        hyps.append(hyp)
        doc_ids.append(doc_id)
    
    # Compute BERTScore
    print("Computing BERTScore (this may take a while)...")
    P, R, F1 = score(hyps, refs, lang='en', verbose=True, device='cpu')
    
    # Create results
    rows = []
    for i, doc_id in enumerate(doc_ids):
        rows.append({
            'doc_id': doc_id,
            'bertscore_precision': float(P[i]),
            'bertscore_recall': float(R[i]),
            'bertscore_f1': float(F1[i])
        })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"BERTScore results saved to {args.out}")
    print(f"Mean F1: {df['bertscore_f1'].mean():.4f}")
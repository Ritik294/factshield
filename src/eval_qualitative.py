"""
Qualitative evaluation: sample best/worst summaries and extract examples.
"""
import argparse
import json
import pandas as pd
from src.utils import read_jsonl
from src.cite import extract_citations


def sample_summaries(rouge_path: str, summary_path: str, nli_path: str, 
                     n_samples: int = 5, out_path: str = None):
    """
    Sample best and worst summaries based on ROUGE and factuality scores.
    
    Args:
        rouge_path: Path to ROUGE CSV
        summary_path: Path to summary JSONL
        nli_path: Path to NLI scores JSONL
        n_samples: Number of samples per category
        out_path: Output JSON path
    """
    # Load data
    rouge_df = pd.read_csv(rouge_path)
    summaries = {r['doc_id']: r['summary'] for r in read_jsonl(summary_path)}
    nli_scores = {r['doc_id']: r['support_ratio'] for r in read_jsonl(nli_path)}
    
    # Combine metrics
    rouge_df['factuality'] = rouge_df['doc_id'].map(nli_scores).fillna(0.0)
    rouge_df['combined_score'] = (
        rouge_df['rouge1'] * 0.4 + 
        rouge_df['rouge2'] * 0.3 + 
        rouge_df['rougeL'] * 0.3 +
        rouge_df['factuality'] * 0.2
    )
    
    # Sample best and worst
    best = rouge_df.nlargest(n_samples, 'combined_score')
    worst = rouge_df.nsmallest(n_samples, 'combined_score')
    
    # Extract examples
    examples = {
        'best': [],
        'worst': []
    }
    
    for idx, row in best.iterrows():
        doc_id = row['doc_id']
        summary = summaries.get(doc_id, '')
        citations = extract_citations(summary)
        examples['best'].append({
            'doc_id': doc_id,
            'rouge1': float(row['rouge1']),
            'rouge2': float(row['rouge2']),
            'rougeL': float(row['rougeL']),
            'factuality': float(row['factuality']),
            'summary': summary,
            'citation_count': len(citations),
            'citations': [f"{c['doc']}:{c['chunk']}" for c in citations]
        })
    
    for idx, row in worst.iterrows():
        doc_id = row['doc_id']
        summary = summaries.get(doc_id, '')
        citations = extract_citations(summary)
        examples['worst'].append({
            'doc_id': doc_id,
            'rouge1': float(row['rouge1']),
            'rouge2': float(row['rouge2']),
            'rougeL': float(row['rougeL']),
            'factuality': float(row['factuality']),
            'summary': summary,
            'citation_count': len(citations),
            'citations': [f"{c['doc']}:{c['chunk']}" for c in citations]
        })
    
    # Save
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        print(f"Qualitative examples saved to {out_path}")
    
    return examples


def compare_citation_quality(summary_path: str, nli_path: str, out_path: str):
    """
    Compare citation quality across summaries.
    
    Args:
        summary_path: Path to summary JSONL
        nli_path: Path to NLI scores JSONL
        out_path: Output CSV path
    """
    summaries = {r['doc_id']: r['summary'] for r in read_jsonl(summary_path)}
    nli_data = {r['doc_id']: r for r in read_jsonl(nli_path)}
    
    rows = []
    for doc_id, summary in summaries.items():
        citations = extract_citations(summary)
        nli_info = nli_data.get(doc_id, {})
        
        # Count citations per sentence
        sentences = summary.split('.')
        cited_sentences = sum(1 for s in sentences if extract_citations(s))
        
        # Citation coverage
        total_sentences = len([s for s in sentences if s.strip()])
        citation_coverage = cited_sentences / max(1, total_sentences)
        
        # Citation support (from NLI)
        support_ratio = nli_info.get('support_ratio', 0.0)
        
        rows.append({
            'doc_id': doc_id,
            'total_citations': len(citations),
            'cited_sentences': cited_sentences,
            'total_sentences': total_sentences,
            'citation_coverage': citation_coverage,
            'support_ratio': support_ratio,
            'citation_quality': support_ratio * citation_coverage  # Combined metric
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Citation quality analysis saved to {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['sample', 'citations'])
    ap.add_argument('--rouge', help='ROUGE CSV path (for sample mode)')
    ap.add_argument('--summ', required=True, help='Summary JSONL path')
    ap.add_argument('--nli', required=True, help='NLI scores JSONL path')
    ap.add_argument('--out', required=True, help='Output path')
    ap.add_argument('--n_samples', type=int, default=5, help='Samples per category')
    args = ap.parse_args()
    
    if args.mode == 'sample':
        sample_summaries(args.rouge, args.summ, args.nli, args.n_samples, args.out)
    elif args.mode == 'citations':
        compare_citation_quality(args.summ, args.nli, args.out)
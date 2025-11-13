"""
Limitations analysis: failure cases, citation failures, length analysis.
"""
import argparse
import pandas as pd
import numpy as np
from src.utils import read_jsonl
from src.cite import extract_citations

def analyze_limitations(nli_path: str, summary_path: str, chunks_path: str, out_path: str):
    """
    Analyze system limitations and failure cases.
    
    Args:
        nli_path: Path to NLI scores JSONL
        summary_path: Path to summaries JSONL
        chunks_path: Path to chunks JSONL
        out_path: Output CSV path
    """
    # Load data
    summaries = {r['doc_id']: r['summary'] for r in read_jsonl(summary_path)}
    nli_data = {r['doc_id']: r for r in read_jsonl(nli_path)}
    chunk_map = {r['chunk_id']: r['text'] for r in read_jsonl(chunks_path)}
    
    rows = []
    
    for doc_id, summary in summaries.items():
        nli_info = nli_data.get(doc_id, {})
        
        # Document length
        doc_length = len(summary.split())
        
        # Citation statistics
        citations = extract_citations(summary)
        citation_count = len(citations)
        
        # Citation failures (citations that don't match chunks)
        citation_failures = 0
        for cit in citations:
            chunk_id = cit['chunk']
            if chunk_id not in chunk_map and f"{doc_id}_{chunk_id}" not in chunk_map:
                citation_failures += 1
        
        # Support ratio
        support_ratio = nli_info.get('support_ratio', 0.0)
        
        # Sentences without citations
        sentences = summary.split('.')
        uncited_sentences = sum(1 for s in sentences if not extract_citations(s))
        
        # Failure category
        if support_ratio == 0.0:
            failure_type = 'no_support'
        elif citation_failures > 0:
            failure_type = 'citation_mismatch'
        elif uncited_sentences > len(sentences) * 0.5:
            failure_type = 'low_citation_coverage'
        else:
            failure_type = 'partial_success'
        
        rows.append({
            'doc_id': doc_id,
            'doc_length_words': doc_length,
            'citation_count': citation_count,
            'citation_failures': citation_failures,
            'uncited_sentences': uncited_sentences,
            'support_ratio': support_ratio,
            'failure_type': failure_type
        })
    
    df = pd.DataFrame(rows)
    
    # Analyze by length bins
    df['length_bin'] = pd.cut(df['doc_length_words'], bins=[0, 100, 500, 1000, float('inf')], 
                              labels=['short', 'medium', 'long', 'very_long'])
    
    # Summary statistics
    summary_stats = {
        'total_docs': len(df),
        'avg_support_ratio': df['support_ratio'].mean(),
        'citation_failure_rate': (df['citation_failures'] > 0).mean(),
        'low_coverage_rate': (df['uncited_sentences'] > df['doc_length_words'] * 0.5).mean(),
        'by_length': df.groupby('length_bin')['support_ratio'].mean().to_dict(),
        'by_failure_type': df['failure_type'].value_counts().to_dict()
    }
    
    # Save
    df.to_csv(out_path, index=False)
    
    summary_path = out_path.replace('.csv', '_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Limitations analysis saved to {out_path}")
    print(f"Summary statistics saved to {summary_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nli', required=True, help='NLI scores JSONL path')
    ap.add_argument('--summ', required=True, help='Summary JSONL path')
    ap.add_argument('--chunks', required=True, help='Chunks JSONL path')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()
    analyze_limitations(args.nli, args.summ, args.chunks, args.out)
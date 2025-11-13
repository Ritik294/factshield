"""
Citation quality evaluation: accuracy, coverage, precision/recall.
"""
import argparse
import pandas as pd
from src.utils import read_jsonl
from src.cite import extract_citations
from src.summarizers import sent_split

def evaluate_citation_quality(summary_path: str, nli_path: str, out_path: str):
    """
    Evaluate citation quality metrics.
    
    Args:
        summary_path: Path to summary JSONL
        nli_path: Path to NLI scores JSONL
        out_path: Output CSV path
    """
    summaries = {r['doc_id']: r['summary'] for r in read_jsonl(summary_path)}
    nli_data = {r['doc_id']: r for r in read_jsonl(nli_path)}
    
    rows = []
    
    for doc_id, summary in summaries.items():
        nli_info = nli_data.get(doc_id, {})
        sentences = sent_split(summary)
        
        # Citation coverage
        cited_sentences = sum(1 for s in sentences if extract_citations(s))
        total_sentences = len(sentences)
        coverage = cited_sentences / max(1, total_sentences)
        
        # Citation accuracy (sentences with citations that are supported)
        supported_cited = sum(1 for s_info in nli_info.get('sentences', [])
                            if s_info.get('status') == 'SUPPORTED' and s_info.get('citations'))
        total_cited = sum(1 for s_info in nli_info.get('sentences', [])
                         if s_info.get('citations'))
        accuracy = supported_cited / max(1, total_cited)
        
        # Citation precision (supported citations / all citations)
        all_citations = extract_citations(summary)
        precision = supported_cited / max(1, len(all_citations))
        
        # Citation recall (cited sentences / total sentences)
        recall = cited_sentences / max(1, total_sentences)
        
        rows.append({
            'doc_id': doc_id,
            'citation_coverage': coverage,
            'citation_accuracy': accuracy,
            'citation_precision': precision,
            'citation_recall': recall,
            'total_citations': len(all_citations),
            'supported_citations': supported_cited
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    
    # Summary statistics
    print(f"Citation quality analysis saved to {out_path}")
    print(f"Mean coverage: {df['citation_coverage'].mean():.3f}")
    print(f"Mean accuracy: {df['citation_accuracy'].mean():.3f}")
    print(f"Mean precision: {df['citation_precision'].mean():.3f}")
    print(f"Mean recall: {df['citation_recall'].mean():.3f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summ', required=True, help='Summary JSONL path')
    ap.add_argument('--nli', required=True, help='NLI scores JSONL path')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()
    evaluate_citation_quality(args.summ, args.nli, args.out)
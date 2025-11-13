"""
Error analysis for NLI verification failures.

Categorizes errors by type (entity, numeric, predicate, discourse) and
generates error distribution statistics across methods.
"""
import argparse
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from src.utils import read_jsonl
from src.summarizers import sent_split


def categorize_error(sentence: str, evidence: str, nli_score: float) -> str:
    """
    Categorize error type based on sentence content and NLI score.
    
    Args:
        sentence: The summary sentence
        evidence: The cited chunk text
        nli_score: NLI margin score (entailment - contradiction)
    
    Returns:
        Error category: 'entity', 'numeric', 'predicate', 'discourse', 'unverifiable', 'other'
    """
    sentence_lower = sentence.lower()
    
    # Numeric errors: numbers, dates, percentages
    import re
    if re.search(r'\d+', sentence) and nli_score < 0:
        return 'numeric'
    
    # Entity errors: proper nouns, organizations
    if any(word in sentence_lower for word in ['department', 'agency', 'congress', 'federal', 'state']):
        if nli_score < 0:
            return 'entity'
    
    # Predicate errors: action verbs, relationships
    action_verbs = ['implemented', 'established', 'created', 'required', 'provided', 'issued']
    if any(verb in sentence_lower for verb in action_verbs) and nli_score < 0:
        return 'predicate'
    
    # Discourse errors: logical connections
    discourse_markers = ['however', 'therefore', 'although', 'because', 'since', 'while']
    if any(marker in sentence_lower for marker in discourse_markers) and nli_score < 0:
        return 'discourse'
    
    # Unverifiable: vague statements
    vague_words = ['some', 'many', 'various', 'several', 'generally', 'often']
    if any(word in sentence_lower for word in vague_words) and nli_score < 0:
        return 'unverifiable'
    
    if nli_score < 0:
        return 'other'
    return None


def analyze_errors(nli_path: str, chunks_path: str, out_path: str):
    """
    Analyze error types from NLI verification results.
    
    Args:
        nli_path: Path to nli_scores.jsonl
        chunks_path: Path to chunks.jsonl for evidence lookup
        out_path: Output CSV path
    """
    # Load chunk map
    chunk_map = {}
    for r in read_jsonl(chunks_path):
        chunk_map[r['chunk_id']] = r['text']
    
    # Analyze errors
    error_counts = defaultdict(int)
    error_by_doc = defaultdict(lambda: defaultdict(int))
    all_errors = []
    
    for row in read_jsonl(nli_path):
        doc_id = row['doc_id']
        for sent_info in row['sentences']:
            if sent_info['status'] == 'LOW-CONF' and sent_info['citations']:
                # Get evidence
                evidence_parts = []
                for cit in sent_info['citations']:
                    chunk_id = cit['chunk']
                    # Try different formats
                    evidence = chunk_map.get(chunk_id, '')
                    if not evidence and doc_id:
                        alt_id = f"{doc_id}_{chunk_id}"
                        evidence = chunk_map.get(alt_id, '')
                    if evidence:
                        evidence_parts.append(evidence)
                
                evidence_text = "\n".join(evidence_parts)
                error_type = categorize_error(sent_info['text'], evidence_text, sent_info['score'])
                
                if error_type:
                    error_counts[error_type] += 1
                    error_by_doc[doc_id][error_type] += 1
                    all_errors.append({
                        'doc_id': doc_id,
                        'sentence': sent_info['text'],
                        'error_type': error_type,
                        'nli_score': sent_info['score'],
                        'has_citation': len(sent_info['citations']) > 0
                    })
    
    # Create summary
    total_errors = sum(error_counts.values())
    summary_rows = []
    for error_type, count in error_counts.items():
        summary_rows.append({
            'error_type': error_type,
            'count': count,
            'percentage': (count / total_errors * 100) if total_errors > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('count', ascending=False)
    
    # Save results
    summary_df.to_csv(out_path, index=False)
    
    # Also save detailed errors
    if all_errors:
        detail_path = out_path.replace('.csv', '_detailed.csv')
        pd.DataFrame(all_errors).to_csv(detail_path, index=False)
    
    print(f"Total errors analyzed: {total_errors}")
    print(f"Error distribution saved to {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nli', required=True, help='Path to nli_scores.jsonl')
    ap.add_argument('--chunks', required=True, help='Path to chunks.jsonl')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()
    
    analyze_errors(args.nli, args.chunks, args.out)
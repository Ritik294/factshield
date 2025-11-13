"""
Bias analysis: gender, entity, and demographic term frequency analysis.
"""
import argparse
import re
import pandas as pd
from collections import Counter
from src.utils import read_jsonl

GENDER_TERMS = {
    'male': ['he', 'him', 'his', 'man', 'men', 'male', 'males', 'gentleman', 'gentlemen'],
    'female': ['she', 'her', 'hers', 'woman', 'women', 'female', 'females', 'lady', 'ladies']
}

DEMOGRAPHIC_TERMS = ['race', 'ethnic', 'minority', 'immigrant', 'refugee', 'citizen', 'alien']

def analyze_bias(summary_path: str, out_path: str):
    """
    Analyze gender and demographic bias in summaries.
    
    Args:
        summary_path: Path to summary JSONL
        out_path: Output CSV path
    """
    gender_counts = Counter()
    demographic_counts = Counter()
    total_words = 0
    
    for row in read_jsonl(summary_path):
        summary = row['summary'].lower()
        words = summary.split()
        total_words += len(words)
        
        # Count gender terms
        for gender, terms in GENDER_TERMS.items():
            for term in terms:
                count = summary.count(term)
                gender_counts[gender] += count
        
        # Count demographic terms
        for term in DEMOGRAPHIC_TERMS:
            count = summary.count(term)
            if count > 0:
                demographic_counts[term] += count
    
    # Create report
    rows = []
    
    # Gender analysis
    total_gender = sum(gender_counts.values())
    for gender, count in gender_counts.items():
        rows.append({
            'category': 'gender',
            'term': gender,
            'count': count,
            'frequency': count / max(1, total_gender) if total_gender > 0 else 0,
            'per_1000_words': (count / max(1, total_words)) * 1000
        })
    
    # Demographic analysis
    for term, count in demographic_counts.items():
        rows.append({
            'category': 'demographic',
            'term': term,
            'count': count,
            'frequency': 0,  # Not applicable
            'per_1000_words': (count / max(1, total_words)) * 1000
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Bias analysis saved to {out_path}")
    print(f"Total words analyzed: {total_words}")
    print(f"Gender term ratio (male/female): {gender_counts['male']/max(1, gender_counts['female']):.2f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--summ', required=True, help='Summary JSONL path')
    ap.add_argument('--out', required=True, help='Output CSV path')
    args = ap.parse_args()
    analyze_bias(args.summ, args.out)
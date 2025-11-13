"""
Enhanced evaluation statistics with statistical tests and per-document breakdowns.
"""
import argparse, glob, pandas as pd
import numpy as np
from scipy import stats

def ci(x):
    """Bootstrap 95% confidence interval."""
    x = np.array(x)
    samples = np.random.default_rng(0).choice(x, size=(1000, len(x)), replace=True)
    lows = np.percentile(samples.mean(axis=1), 2.5)
    highs = np.percentile(samples.mean(axis=1), 97.5)
    return lows, highs

def statistical_test(group1, group2):
    """Perform t-test between two groups."""
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='indir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per_doc', action='store_true', help='Generate per-document breakdown')
    args = ap.parse_args()

    # Load all ROUGE files
    import os
    dfs = {}
    for fp in glob.glob(args.indir + '/*.csv'):
        if 'rouge' in fp.lower():
            df = pd.read_csv(fp)
            # Handle both Windows and Unix paths
            basename = os.path.basename(fp)
            model = basename.replace('rouge_', '').replace('.csv', '')
            # Remove 'eval' prefix if present
            if model.startswith('eval'):
                model = model.replace('eval', '').replace('\\', '').replace('/', '').strip('_')
            dfs[model] = df

     # Aggregate statistics
    rows = []
    for model, df in dfs.items():
        for m in ['rouge1','rouge2','rougeL']:
            lo, hi = ci(df[m].values)
            rows.append({
                'model': model,
                'metric': m,
                'mean': df[m].mean(),
                'std': df[m].std(),
                'ci_low': lo,
                'ci_high': hi,
                'n': len(df)
            })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.out, index=False)
    
    # Statistical tests (compare RAG vs baselines)
    if 'rag' in dfs and len(dfs) > 1:
        test_results = []
        rag_df = dfs['rag']
        for baseline in ['textrank', 'distilbart', 'lead3']:  # Added lead3
            if baseline in dfs:
                baseline_df = dfs[baseline]
                for metric in ['rouge1', 'rouge2', 'rougeL']:
                    test = statistical_test(rag_df[metric].values, baseline_df[metric].values)
                    test_results.append({
                        'comparison': f'rag_vs_{baseline}',
                        'metric': metric,
                        **test
                    })
        
        if test_results:
            test_df = pd.DataFrame(test_results)
            test_path = args.out.replace('.csv', '_statistical_tests.csv')
            test_df.to_csv(test_path, index=False)
            print(f"Statistical tests saved to {test_path}")
    # Per-document breakdown
    if args.per_doc:
        per_doc_rows = []
        for model, df in dfs.items():
            for _, row in df.iterrows():
                per_doc_rows.append({
                    'doc_id': row['doc_id'],
                    'model': model,
                    'rouge1': row['rouge1'],
                    'rouge2': row['rouge2'],
                    'rougeL': row['rougeL']
                })
        per_doc_df = pd.DataFrame(per_doc_rows)
        per_doc_path = args.out.replace('.csv', '_per_doc.csv')
        per_doc_df.to_csv(per_doc_path, index=False)
        print(f"Per-document breakdown saved to {per_doc_path}")
    
    print(f"Summary statistics saved to {args.out}")
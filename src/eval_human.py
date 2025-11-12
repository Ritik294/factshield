import argparse, json, pandas as pd
from sklearn.metrics import cohen_kappa_score


# Human eval schema: CSV with columns
# doc_id, sent_id, text, label_annotator1, label_annotator2, error_type
# label_* in {SUPPORTED, LOW-CONF}; error_type from FRANK (free text / enum)


if __name__ == '__main__':
ap = argparse.ArgumentParser()
ap.add_argument('--csv', required=True) # annotations.csv
ap.add_argument('--out', required=True) # human_eval_summary.json
args = ap.parse_args()


df = pd.read_csv(args.csv)
kappa = cohen_kappa_score(df['label_annotator1'], df['label_annotator2'])
# simple tallies
counts = df['error_type'].value_counts().to_dict()
agree = (df['label_annotator1'] == df['label_annotator2']).mean()
out = {
'n_items': int(len(df)),
'agreement_rate': float(agree),
'cohen_kappa': float(kappa),
'error_type_counts': counts
}
with open(args.out, 'w', encoding='utf-8') as f:
json.dump(out, f, ensure_ascii=False, indent=2)
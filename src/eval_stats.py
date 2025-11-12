import argparse, glob, pandas as pd
import numpy as np


# Aggregate CSVs, compute bootstrap 95% CIs


def ci(x):
x = np.array(x)
samples = np.random.default_rng(0).choice(x, size=(1000, len(x)), replace=True)
lows = np.percentile(samples.mean(axis=1), 2.5)
highs = np.percentile(samples.mean(axis=1), 97.5)
return lows, highs


if __name__ == '__main__':
ap = argparse.ArgumentParser()
ap.add_argument('--in', dest='indir', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()


rows = []
for fp in glob.glob(args.indir + '/*.csv'):
df = pd.read_csv(fp)
model = fp.split('/')[-1].replace('.csv','')
for m in ['rouge1','rouge2','rougeL']:
lo, hi = ci(df[m].values)
rows.append({'model': model, 'metric': m, 'mean': df[m].mean(), 'ci_low': lo, 'ci_high': hi})
out = pd.DataFrame(rows)
out.to_csv(args.out, index=False)
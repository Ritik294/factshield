import argparse, math
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from src.utils import read_jsonl, write_jsonl
from src.cite import extract_citations


# NLI verifier: sentence vs concatenated cited chunks → entail/neutral/contradict


def nli_scores(model, tokenizer, premise: str, hypothesis: str):
inputs = tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors='pt')
with torch.no_grad():
logits = model(**inputs).logits[0]
probs = torch.softmax(logits, dim=-1).tolist() # [contradiction, neutral, entailment]
return {'c': probs[0], 'n': probs[1], 'e': probs[2], 'margin': probs[2]-probs[0]}


if __name__ == '__main__':
ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['nli'])
ap.add_argument('--summ', required=True) # runs/improved/rag_cited.jsonl
ap.add_argument('--chunks', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--threshold', type=float, default=0.2)
args = ap.parse_args()


tok = AutoTokenizer.from_pretrained('roberta-base-mnli')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-mnli')


# chunk lookup
chunk_map = {}
for r in read_jsonl(args.chunks):
chunk_map[r['chunk_id']] = r['text']


rows = []
for row in read_jsonl(args.summ):
doc_id = row['doc_id']
summary = row['summary']
sents = [s.strip() for s in summary.split('.') if s.strip()]
out_sents = []
for s in sents:
cits = extract_citations(s)
evidence = "\n".join([chunk_map.get(c['chunk'], '') for c in cits])
if not evidence:
out_sents.append({'text': s, 'status': 'LOW-CONF', 'score': 0.0, 'citations': cits})
continue
sc = nli_scores(model, tok, evidence, s)
status = 'SUPPORTED' if (sc['margin'] >= args.threshold and sc['e']>sc['c']) else 'LOW-CONF'
out_sents.append({'text': s, 'status': status, 'score': sc['margin'], 'citations': cits})
support = sum(1 for x in out_sents if x['status']=='SUPPORTED')/max(1,len(out_sents))
rows.append({'doc_id': doc_id, 'sentences': out_sents, 'support_ratio': support})
write_jsonl(args.out, rows)
import argparse, math
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from src.utils import read_jsonl, write_jsonl
from src.cite import extract_citations
from src.summarizers import sent_split

# NLI verifier: sentence vs concatenated cited chunks → entail/neutral/contradict

def nli_scores(model, tokenizer, premise: str, hypothesis: str):
    inputs = tokenizer(premise, hypothesis, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()  # [contradiction, neutral, entailment]
    return {'c': probs[0], 'n': probs[1], 'e': probs[2], 'margin': probs[2]-probs[0]}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['nli'])
    ap.add_argument('--summ', required=True)  # runs/improved/rag_cited.jsonl
    ap.add_argument('--chunks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', type=float, default=0.1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained('roberta-large-mnli')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')

    # chunk lookup
    chunk_map = {}
    for r in read_jsonl(args.chunks):
        chunk_map[r['chunk_id']] = r['text']

    rows = []
    for row in read_jsonl(args.summ):
        doc_id = row['doc_id']
        summary = row['summary']
        sents = sent_split(summary)
        out_sents = []
        for s in sents:
            # Extract citations BEFORE cleaning text
            cits = extract_citations(s)
            
            # Remove citations from sentence text for NLI (keep only the actual claim)
            import re
            clean_text = re.sub(r'\[CIT:[^\]]+\]', '', s).strip()
            
            # Skip if sentence is just a citation tag or empty
            if not clean_text or clean_text == '':
                continue
            
            # Try to find chunks - handle different citation formats
            evidence_parts = []
            for cit in cits:
                chunk_id = cit['chunk']
                # Try exact match first
                chunk_text = chunk_map.get(chunk_id, '')
                # If not found, try with doc_id prefix
                if not chunk_text and doc_id:
                    alt_id = f"{doc_id}_{chunk_id}"
                    chunk_text = chunk_map.get(alt_id, '')
                # If still not found, try removing doc_id prefix if present
                if not chunk_text and chunk_id.startswith(f"{doc_id}_"):
                    alt_id = chunk_id.replace(f"{doc_id}_", "", 1)
                    chunk_text = chunk_map.get(alt_id, '')
                if chunk_text:
                    evidence_parts.append(chunk_text)
            
            evidence = "\n".join(evidence_parts)
            
            if not evidence:
                out_sents.append({'text': clean_text, 'status': 'LOW-CONF', 'score': 0.0, 'citations': cits})
                continue
            
            # Use clean text (without citations) for NLI
            sc = nli_scores(model, tok, evidence, clean_text)
            status = 'SUPPORTED' if (sc['margin'] >= args.threshold and sc['e']>sc['c']) else 'LOW-CONF'
            out_sents.append({'text': clean_text, 'status': status, 'score': sc['margin'], 'citations': cits})
        
        support = sum(1 for x in out_sents if x['status']=='SUPPORTED')/max(1,len(out_sents))
        rows.append({'doc_id': doc_id, 'sentences': out_sents, 'support_ratio': support})
    write_jsonl(args.out, rows)
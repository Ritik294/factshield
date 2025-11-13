"""
Ablation studies: test different hyperparameters and configurations.
"""
import argparse
import json
import pandas as pd
import numpy as np
from src.utils import read_jsonl, write_jsonl
from src.chunker import chunk_text
from transformers import AutoTokenizer
import os
import glob

def test_topk(topk_values: list, base_chunks: str, base_meta: str, 
              base_index: str, base_summaries: str, out_path: str):
    """
    Test different top-k retrieval values.
    
    Args:
        topk_values: List of k values to test (e.g., [1, 3, 5, 10])
        base_chunks: Path to chunks.jsonl
        base_meta: Path to meta.json
        base_index: Path to FAISS index
        base_summaries: Not used (for compatibility)
        out_path: Output CSV path
    """
    from src.embed_retrieve import retrieve_topk
    from src.summarizers import run_rag
    
    results = []
    
    for k in topk_values:
        print(f"Testing top_k={k}...")
        # Retrieve with different k
        topk_path = f"cache/retrieval/topk_{k}.jsonl"
        os.makedirs(os.path.dirname(topk_path), exist_ok=True)
        retrieve_topk(base_chunks, base_meta, base_index, k, topk_path)
        
        # Generate summaries
        summ_path = f"runs/ablation/rag_k{k}.jsonl"
        os.makedirs(os.path.dirname(summ_path), exist_ok=True)
        run_rag(base_chunks, topk_path, summ_path)
        
        # Verify using verifier logic directly (avoid argparse issues)
        nli_path = f"runs/ablation/nli_k{k}.jsonl"
        os.makedirs(os.path.dirname(nli_path), exist_ok=True)
        
        # Import verifier components
        from src.verifier import nli_scores
        from src.summarizers import sent_split
        from src.cite import extract_citations
        from transformers import AutoTokenizer as AT, AutoModelForSequenceClassification
        import torch
        import re
        
        # Load NLI model
        nli_tok = AT.from_pretrained('roberta-large-mnli')
        nli_model = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
        
        # Load chunk map
        chunk_map = {}
        for r in read_jsonl(base_chunks):
            chunk_map[r['chunk_id']] = r['text']
        
        # Verify summaries
        rows = []
        for row in read_jsonl(summ_path):
            doc_id = row['doc_id']
            summary = row['summary']
            sents = sent_split(summary)
            out_sents = []
            for s in sents:
                cits = extract_citations(s)
                clean_text = re.sub(r'\[CIT:[^\]]+\]', '', s).strip()
                if not clean_text:
                    continue
                
                evidence_parts = []
                for cit in cits:
                    chunk_id = cit['chunk']
                    chunk_text = chunk_map.get(chunk_id, '')
                    if not chunk_text and doc_id:
                        alt_id = f"{doc_id}_{chunk_id}"
                        chunk_text = chunk_map.get(alt_id, '')
                    if chunk_text:
                        evidence_parts.append(chunk_text)
                
                evidence = "\n".join(evidence_parts)
                if not evidence:
                    out_sents.append({'text': clean_text, 'status': 'LOW-CONF', 'score': 0.0, 'citations': cits})
                    continue
                
                sc = nli_scores(nli_model, nli_tok, evidence, clean_text)
                status = 'SUPPORTED' if (sc['margin'] >= 0.1 and sc['e']>sc['c']) else 'LOW-CONF'
                out_sents.append({'text': clean_text, 'status': status, 'score': sc['margin'], 'citations': cits})
            
            support = sum(1 for x in out_sents if x['status']=='SUPPORTED')/max(1,len(out_sents))
            rows.append({'doc_id': doc_id, 'sentences': out_sents, 'support_ratio': support})
        
        write_jsonl(nli_path, rows)
        
        # Compute factuality
        support_ratios = []
        for r in read_jsonl(nli_path):
            support_ratios.append(r['support_ratio'])
        
        results.append({
            'top_k': k,
            'mean_support_ratio': np.mean(support_ratios),
            'std_support_ratio': np.std(support_ratios)
        })
    
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"Top-k ablation results saved to {out_path}")

def test_chunksize(chunk_sizes: list, input_dir: str, base_emb_model: str, out_path: str):
    """
    Test different chunk sizes.
    
    Args:
        chunk_sizes: List of chunk token sizes to test (e.g., [500, 900, 1200])
        input_dir: Directory containing input documents
        base_emb_model: Not used (for compatibility)
        out_path: Output CSV path
    """
    tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    
    results = []
    
    for size in chunk_sizes:
        print(f"Testing chunk_size={size}...")
        # Chunk with different size
        chunks_path = f"cache/chunks/chunks_{size}.jsonl"
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(input_dir, '*')))
        rows = []
        for fp in files:
            if not os.path.isfile(fp):
                continue
            doc_id = os.path.splitext(os.path.basename(fp))[0]
            with open(fp, 'r', encoding='utf-8') as f:
                text = f.read()
            for i, (s, e, piece) in enumerate(chunk_text(text, tok, chunk_tokens=size, stride=200)):
                rows.append({
                    'doc_id': doc_id,
                    'chunk_id': f"{doc_id}_{i:04d}",
                    'start_token': s,
                    'end_token': e,
                    'text': piece
                })
        write_jsonl(chunks_path, rows)
        
        # Count chunks per doc
        by_doc = {}
        for r in read_jsonl(chunks_path):
            by_doc.setdefault(r['doc_id'], []).append(r)
        
        avg_chunks = np.mean([len(chunks) for chunks in by_doc.values()])
        
        results.append({
            'chunk_size': size,
            'avg_chunks_per_doc': avg_chunks
        })
    
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"Chunk size ablation results saved to {out_path}")

def test_nli_threshold(thresholds: list, nli_path: str, out_path: str):
    """
    Test different NLI thresholds.
    
    Args:
        thresholds: List of threshold values to test (e.g., [0.1, 0.2, 0.3])
        nli_path: Path to existing NLI scores JSONL
        out_path: Output CSV path
    """
    results = []
    
    for thresh in thresholds:
        # Re-verify with different threshold
        support_ratios = []
        for row in read_jsonl(nli_path):
            supported = sum(1 for s in row['sentences'] 
                          if s.get('score', 0) >= thresh and s.get('status') == 'SUPPORTED')
            total = len(row['sentences'])
            support_ratios.append(supported / max(1, total))
        
        results.append({
            'threshold': thresh,
            'mean_support_ratio': np.mean(support_ratios),
            'std_support_ratio': np.std(support_ratios)
        })
    
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"NLI threshold ablation results saved to {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['topk', 'chunksize', 'threshold'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunks', help='Chunks path (for topk)')
    ap.add_argument('--meta', help='Meta path (for topk)')
    ap.add_argument('--index', help='Index path (for topk)')
    ap.add_argument('--nli', help='NLI path (for threshold)')
    ap.add_argument('--input_dir', help='Input dir (for chunksize)')
    args = ap.parse_args()
    
    if args.mode == 'topk':
        test_topk([1, 3, 5, 10], args.chunks, args.meta, args.index, None, args.out)
    elif args.mode == 'chunksize':
        test_chunksize([500, 900, 1200], args.input_dir, None, args.out)
    elif args.mode == 'threshold':
        test_nli_threshold([0.1, 0.2, 0.3], args.nli, args.out)
import argparse, os, json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import spacy
from src.utils import read_jsonl, write_jsonl

NLP = None

def sent_split(text:str):
    global NLP
    if NLP is None:
        NLP = spacy.load('en_core_web_sm', disable=["ner", "tagger", "lemmatizer"])
    return [s.text.strip() for s in NLP(text).sents if s.text.strip()]

# Baseline: TextRank (extractive)

def run_textrank(input_dir, out_path):
    import pytextrank, spacy
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('textrank')

    rows = []
    for fname in sorted(os.listdir(input_dir)):
        fp = os.path.join(input_dir, fname)
        if not os.path.isfile(fp):
            continue
        doc_id = os.path.splitext(fname)[0]
        text = open(fp, 'r', encoding='utf-8').read()
        doc = nlp(text)
        summary = doc._.textrank.summary(limit_sentences=7)
        summ_text = " ".join([s.text for s in summary])
        rows.append({'doc_id': doc_id, 'summary': summ_text})
    write_jsonl(out_path, rows)

# Baseline: DistilBART on first chunk (cheap)

def run_distilbart(chunks_path, out_path):
    tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
    pipe = pipeline('summarization', model=model, tokenizer=tok, device=-1)

    by_doc = {}
    for r in read_jsonl(chunks_path):
        by_doc.setdefault(r['doc_id'], []).append(r)

    rows = []
    for doc_id, chunks in tqdm(by_doc.items()):
        first = chunks[0]['text'][:1024]
        out = pipe(first, max_new_tokens=180, min_new_tokens=60, num_beams=4, no_repeat_ngram_size=3)[0]['summary_text']
        rows.append({'doc_id': doc_id, 'summary': out})
    write_jsonl(out_path, rows)

# Improved: RAG + must‑cite, summarize concatenated top‑k chunks

def run_rag(chunks_path, topk_path, out_path):
    tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
    pipe = pipeline('summarization', model=model, tokenizer=tok, device=-1)

    # Map chunk_id → text
    chunk_map = {}
    for r in read_jsonl(chunks_path):
        chunk_map[r['chunk_id']] = r['text']

    rows = []
    for row in read_jsonl(topk_path):
        doc_id = row['doc_id']
        top = row['topk']
        # concatenate truncated chunks with labels
        contexts = []
        for meta in top:
            cid = meta['chunk_id']
            ctx = (f"[CIT:{doc_id}:{cid}]\n" + chunk_map[cid][:900])
            contexts.append(ctx)
        joined = "\n\n".join(contexts)[:2500]
        prompt = (
            "Summarize the document faithfully. For each factual statement, append the citation tag of the supporting chunk in square brackets, e.g., [CIT:doc:chunk].\n\n" + joined
        )
        out = pipe(prompt, max_new_tokens=220, min_new_tokens=80, num_beams=4, no_repeat_ngram_size=3)[0]['summary_text']
        rows.append({'doc_id': doc_id, 'summary': out})
    write_jsonl(out_path, rows)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['textrank','distilbart','rag'])
    ap.add_argument('--input_dir')
    ap.add_argument('--chunks')
    ap.add_argument('--topk')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.mode == 'textrank':
        run_textrank(args.input_dir, args.out)
    elif args.mode == 'distilbart':
        run_distilbart(args.chunks, args.out)
    elif args.mode == 'rag':
        run_rag(args.chunks, args.topk, args.out)
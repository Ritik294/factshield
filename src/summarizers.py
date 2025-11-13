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

# Baseline: Lead-3 (first 3 sentences - strong extractive baseline)

def run_lead3(input_dir, out_path):
    """
    Extract first 3 sentences as summary (strong extractive baseline).
    
    Args:
        input_dir: Directory containing input documents
        out_path: Output JSONL path
    """
    rows = []
    for fname in sorted(os.listdir(input_dir)):
        fp = os.path.join(input_dir, fname)
        if not os.path.isfile(fp):
            continue
        doc_id = os.path.splitext(fname)[0]
        text = open(fp, 'r', encoding='utf-8').read()
        sents = sent_split(text)
        summary = " ".join(sents[:3]) if len(sents) >= 3 else " ".join(sents)
        rows.append({'doc_id': doc_id, 'summary': summary})
    write_jsonl(out_path, rows)

def run_rag(chunks_path, topk_path, out_path):
    from src.seed import set_seed
    set_seed(42)
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
            # Ensure chunk_id format is consistent (use as-is from topk)
            ctx = (f"[CIT:{doc_id}:{cid}]\n" + chunk_map[cid][:900])
            contexts.append(ctx)
        joined = "\n\n".join(contexts)[:2500]
        
        # IMPROVED PROMPT: More explicit instruction
        prompt = (
            f"Document sections:\n{joined}\n\n"
            f"Task: Write a summary that cites sources. After each fact, add [CIT:{doc_id}:CHUNK_ID] where CHUNK_ID matches the source.\n"
            f"Example: 'The budget was $50 million [CIT:{doc_id}:chunk_0001].'\n"
            f"Summary:"
        )
        
        out = pipe(prompt, max_new_tokens=220, min_new_tokens=80, num_beams=4, no_repeat_ngram_size=3)[0]['summary_text']
        
        # POST-PROCESSING: If model still doesn't cite, add citations based on sentence similarity
        # This is a fallback to ensure citations exist
        from src.cite import extract_citations
        if not extract_citations(out):
            # If no citations found, try to add them heuristically
            sentences = sent_split(out)
            cited_sentences = []
            for sent in sentences:
                # Find best matching chunk for this sentence
                best_chunk = None
                best_score = 0
                for meta in top:
                    cid = meta['chunk_id']
                    chunk_text = chunk_map[cid][:200]  # Compare with chunk start
                    # Simple word overlap
                    sent_words = set(sent.lower().split())
                    chunk_words = set(chunk_text.lower().split())
                    overlap = len(sent_words & chunk_words) / max(len(sent_words), 1)
                    if overlap > best_score and overlap > 0.1:
                        best_score = overlap
                        best_chunk = cid
                
                if best_chunk:
                    cited_sentences.append(f"{sent} [CIT:{doc_id}:{best_chunk}]")
                else:
                    cited_sentences.append(sent)
            out = " ".join(cited_sentences)
        
        rows.append({'doc_id': doc_id, 'summary': out})
    write_jsonl(out_path, rows)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['textrank','distilbart','rag','lead3'])
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
    elif args.mode == 'lead3':
        run_lead3(args.input_dir, args.out)
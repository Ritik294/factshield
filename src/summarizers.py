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
run_rag(args.chunks, args.topk, args.out)
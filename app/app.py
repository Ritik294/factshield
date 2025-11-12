import gradio as gr
import os, json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from src.chunker import chunk_text
from src.verifier import nli_scores
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Minimal CPU demo: upload text → RAG‑ish summary + NLI per sentence


def build_pipes():
summ_tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
summ_model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
summarizer = pipeline('summarization', model=summ_model, tokenizer=summ_tok, device=-1)
emb = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
nli_tok = AutoTokenizer.from_pretrained('roberta-base-mnli')
nli = AutoModelForSequenceClassification.from_pretrained('roberta-base-mnli')
return summarizer, emb, nli_tok, nli


SUMM, EMB, NLI_TOK, NLI = build_pipes()


def summarize(text):
# chunk
chunks = []
for i, (s,e,piece) in enumerate(chunk_text(text, SUMM.tokenizer if hasattr(SUMM, 'tokenizer') else AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6'))):
chunks.append((f"chunk{i:03d}", piece))
# pick top 3 by similarity to middle chunk
mid = len(chunks)//2
q = chunks[mid][1]
qemb = EMB.encode([q], normalize_embeddings=True)
texts = [c[1] for c in chunks]
embs = EMB.encode(texts, normalize_embeddings=True)
import numpy as np
sims = (qemb @ embs.T)[0]
top = np.argsort(-sims)[:3]
ctx = []
for idx in top:
cid, t = chunks[idx]
ctx.append(f"[CIT:doc:{cid}]\n{t[:900]}")
prompt = "Summarize faithfully with citations like [CIT:doc:chunk].\n\n" + "\n\n".join(ctx)
out = SUMM(prompt, max_new_tokens=200, min_new_tokens=80, num_beams=4, no_repeat_ngram_size=3)[0]['summary_text']


# sentence‑level NLI
sents = [s.strip() for s in out.split('.') if s.strip()]
results = []
chunk_map = {cid:t for cid,t in chunks}
import re
for s in sents:
cits = re.findall(r"\[CIT:doc:(chunk\d+)\]", s)
ev = "\n".join([chunk_map.get(c,'') for c in cits])
if not ev:
results.append((s, 'LOW-CONF', 0.0))
else:
sc = nli_scores(NLI, NLI_TOK, ev, s)
label = 'SUPPORTED' if (sc['margin']>=0.2 and sc['e']>sc['c']) else 'LOW-CONF'
results.append((s, label, round(sc['margin'],3)))
formatted = "\n".join([f"- {lab}: {s} (Δ={m})" for s,lab,m in results])
support_ratio = sum(1 for _,lab,_ in results if lab=='SUPPORTED')/max(1,len(results))
return out, f"Factuality support ratio: {support_ratio:.2f}", formatted


with gr.Blocks() as demo:
gr.Markdown("# FactShield — CPU Demo")
inp = gr.Textbox(lines=12, label="Paste long text (e.g., 2–5 pages)")
btn = gr.Button("Summarize with citations + verify")
out_summary = gr.Textbox(lines=10, label="Summary")
out_score = gr.Markdown()
out_detail = gr.Textbox(lines=12, label="Sentence verification")
btn.click(summarize, inputs=inp, outputs=[out_summary, out_score, out_detail])
demo.launch()
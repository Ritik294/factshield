import gradio as gr
import os, json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from src.chunker import chunk_text
from src.verifier import nli_scores
from transformers import AutoModelForSequenceClassification
import re

# Fixed: Use roberta-large-mnli
def build_pipes():
    try:
        summ_tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        summ_model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
        summarizer = pipeline('summarization', model=summ_model, tokenizer=summ_tok, device=-1)
        emb = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        nli_tok = AutoTokenizer.from_pretrained('roberta-large-mnli')  # FIXED
        nli = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')  # FIXED
        return summarizer, emb, nli_tok, nli
    except Exception as e:
        return None, None, None, None

SUMM, EMB, NLI_TOK, NLI = build_pipes()

def summarize(text):
    if not text or len(text.strip()) < 50:
        return "Please provide at least 50 characters of text.", "Error: Input too short", ""
    
    if SUMM is None:
        return "Error: Models not loaded. Please check installation.", "Error", ""
    
    try:
        # Chunk
        chunks = []
        tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        for i, (s, e, piece) in enumerate(chunk_text(text, tok)):
            chunks.append((f"chunk{i:03d}", piece))
        
        if len(chunks) == 0:
            return "Error: Could not chunk text.", "Error", ""
        
        # Retrieve top 3
        mid = len(chunks) // 2
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
        
        # Sentence-level NLI with citation highlighting
        from src.summarizers import sent_split
        sents = sent_split(out)
        results = []
        chunk_map = {cid: t for cid, t in chunks}
        from src.cite import extract_citations
        
        for s in sents:
            cits = extract_citations(s)
            clean_s = re.sub(r'\[CIT:[^\]]+\]', '', s).strip()
            if not clean_s:
                continue
            
            ev = "\n".join([chunk_map.get(c['chunk'], '') for c in cits])
            if not ev:
                results.append((s, 'LOW-CONF', 0.0, []))
            else:
                sc = nli_scores(NLI, NLI_TOK, ev, clean_s)
                label = 'SUPPORTED' if (sc['margin'] >= 0.1 and sc['e'] > sc['c']) else 'LOW-CONF'
                results.append((s, label, round(sc['margin'], 3), cits))
        
        # Format with citation highlighting
        formatted_summary = out
        for s, lab, m, cits in results:
            if cits:
                # Highlight citations in summary
                for cit in cits:
                    cit_tag = f"[CIT:{cit['doc']}:{cit['chunk']}]"
                    formatted_summary = formatted_summary.replace(cit_tag, f"**{cit_tag}**")
        
        formatted_detail = "\n".join([
            f"- **{lab}**: {s} (Δ={m})" + (f" [Citations: {len(cits)}]" if cits else "")
            for s, lab, m, cits in results
        ])
        
        support_ratio = sum(1 for _, lab, _, _ in results if lab == 'SUPPORTED') / max(1, len(results))
        return formatted_summary, f"**Factuality support ratio: {support_ratio:.2%}**", formatted_detail
    
    except Exception as e:
        return f"Error: {str(e)}", "Error", ""

with gr.Blocks(title="FactShield Demo") as demo:
    gr.Markdown("# FactShield — Evidence-Grounded Summarization Demo")
    gr.Markdown("Paste a long document (2-5 pages) to get a summary with citations and factuality verification.")
    
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                lines=15, 
                label="Input Document",
                placeholder="Paste your document here..."
            )
            example_text = """The Department of Defense (DOD) has implemented several cybersecurity initiatives to protect defense small businesses. These efforts include new guidance for cyberspace operations budget submissions and operational risk assessments. However, DOD has not established a timeframe for implementation or identified a process to hold leaders accountable."""
            gr.Examples([[example_text]], inputs=inp)
            btn = gr.Button("Summarize with Citations + Verify", variant="primary")
        
        with gr.Column():
            out_summary = gr.Markdown(label="Summary with Citations")
            out_score = gr.Markdown(label="Factuality Score")
            out_detail = gr.Markdown(label="Sentence Verification Details")
    
    btn.click(
        summarize, 
        inputs=inp, 
        outputs=[out_summary, out_score, out_detail]
    )

if __name__ == '__main__':
    demo.launch(share=True, server_name="0.0.0.0")
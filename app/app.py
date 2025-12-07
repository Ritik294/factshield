import gradio as gr
import os, json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from src.chunker import chunk_text
from src.verifier import nli_scores
from transformers import AutoModelForSequenceClassification
from src.cite import extract_citations
from src.summarizers import sent_split
import re
import numpy as np

def build_pipes():
    try:
        summ_tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        summ_model = AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
        summarizer = pipeline('summarization', model=summ_model, tokenizer=summ_tok, device=-1)
        emb = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        nli_tok = AutoTokenizer.from_pretrained('roberta-large-mnli')
        nli = AutoModelForSequenceClassification.from_pretrained('roberta-large-mnli')
        return summarizer, emb, nli_tok, nli
    except Exception as e:
        return None, None, None, None

SUMM, EMB, NLI_TOK, NLI = build_pipes()

def summarize(text, doc_id="demo"):
    """Summarization with heuristic citation fallback."""
    if not text or len(text.strip()) < 200:  
        return "Please provide at least 200 characters of text (preferably 1000+ for best results).", "Error: Input too short", ""
    
    if SUMM is None:
        return "Error: Models not loaded. Please check installation.", "Error", ""
    
    try:
        # Chunk with proper tokenizer
        chunks = []
        tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        for i, (s, e, piece) in enumerate(chunk_text(text, tok, chunk_tokens=900, stride=200)):
            chunks.append((f"chunk{i:04d}", piece))
        
        if len(chunks) == 0:
            return "Error: Could not chunk text.", "Error", ""
        
        # Build chunk map for citation lookup
        chunk_map = {cid: t for cid, t in chunks}
        
        # Retrieve top-k chunks using embedding similarity
        query_text = text[:500] if len(text) > 500 else text
        qemb = EMB.encode([query_text], normalize_embeddings=True)
        texts = [c[1] for c in chunks]
        embs = EMB.encode(texts, normalize_embeddings=True)
        sims = (qemb @ embs.T)[0]
        top_k = min(5, len(chunks))  # Use top 5 chunks
        top_indices = np.argsort(-sims)[:top_k]
        
        # Build context with citation tags
        contexts = []
        for idx in top_indices:
            cid, chunk_content = chunks[idx]
            contexts.append(f"[CIT:{doc_id}:{cid}]\n{chunk_content[:900]}")
        
        joined = "\n\n".join(contexts)[:2500]
        
        prompt = (
            f"Document sections:\n{joined}\n\n"
            f"Task: Write a summary that cites sources. After each fact, add [CIT:{doc_id}:CHUNK_ID] where CHUNK_ID matches the source.\n"
            f"Example: 'The budget was $50 million [CIT:{doc_id}:chunk_0001].'\n"
            f"Summary:"
        )
        
        out = SUMM(prompt, max_new_tokens=220, min_new_tokens=80, num_beams=4, no_repeat_ngram_size=3)[0]['summary_text']
        
        if not extract_citations(out):
            sentences = sent_split(out)
            cited_sentences = []
            for sent in sentences:
                best_chunk = None
                best_score = 0
                for idx in top_indices:
                    cid, chunk_content = chunks[idx][0], chunks[idx][1]
                    sent_words = set(sent.lower().split())
                    chunk_words = set(chunk_content[:200].lower().split())
                    overlap = len(sent_words & chunk_words) / max(len(sent_words), 1)
                    if overlap > best_score and overlap > 0.1:
                        best_score = overlap
                        best_chunk = cid
                
                if best_chunk:
                    cited_sentences.append(f"{sent} [CIT:{doc_id}:{best_chunk}]")
                else:
                    cited_sentences.append(sent)
            out = " ".join(cited_sentences)
        
        sents = sent_split(out)
        results = []
        all_citations = extract_citations(out)
        citation_map = {}
        
        for s in sents:
            cits = extract_citations(s)
            clean_s = re.sub(r'\[CIT:[^\]]+\]', '', s).strip()
            if not clean_s:
                continue
            
            if not cits:
                if clean_s:
                    for cit in all_citations:
                        cit_pattern = f"\\[CIT:{cit['doc']}:{cit['chunk']}\\]"
                        if re.search(cit_pattern, out):
                            sent_with_cit = re.search(f"{re.escape(clean_s[:50])}.*?{cit_pattern}|{cit_pattern}.*?{re.escape(clean_s[:50])}", out)
                            if sent_with_cit:
                                cits = extract_citations(sent_with_cit.group())
                                break
            
            evidence_parts = []
            for cit in cits:
                chunk_id = cit['chunk']
                chunk_content = chunk_map.get(chunk_id, '')
                if not chunk_content and chunk_id.startswith(f"{doc_id}_"):
                    alt_id = chunk_id.replace(f"{doc_id}_", "", 1)
                    chunk_content = chunk_map.get(alt_id, '')
                
                if not chunk_content:
                    alt_id = chunk_id.replace('_', '')
                    chunk_content = chunk_map.get(alt_id, '')
                
                if chunk_content:
                    evidence_parts.append(chunk_content)
            
            ev = "\n".join(evidence_parts)
            if ev:
                
                ev = ev[:3200] if len(ev) > 3200 else ev
            if not ev:
                results.append((s, 'LOW-CONF', 0.0, []))
            else:
                sc = nli_scores(NLI, NLI_TOK, ev, clean_s)
                label = 'SUPPORTED' if (sc['margin'] >= 0.05 and sc['e'] > sc['c']) else 'LOW-CONF'
                results.append((s, label, round(sc['margin'], 3), cits))
        
        formatted_summary = out
        for s, lab, m, cits in results:
            if cits:
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
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", "Error", ""

def process_file(file):
    """Handle file upload."""
    if file is None:
        return None
    
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        return f"Error reading file: {str(e)}"

EXAMPLE_TEXT = None
if os.path.exists("data/govreport_subset"):
    example_files = sorted([f for f in os.listdir("data/govreport_subset") if f.endswith('.txt')])
    if example_files:
        example_path = os.path.join("data/govreport_subset", example_files[0])
        try:
            with open(example_path, 'r', encoding='utf-8') as f:
                EXAMPLE_TEXT = f.read()[:3000]  
        except:
            pass

if EXAMPLE_TEXT is None:
    EXAMPLE_TEXT = """The Department of Defense (DOD) has implemented several cybersecurity initiatives to protect defense small businesses. These efforts include new guidance for cyberspace operations budget submissions and operational risk assessments. The guidance requires agencies to submit detailed plans and timelines for compliance with cybersecurity standards. However, DOD has not established a clear timeframe for implementation or identified a process to hold leaders accountable for meeting these requirements. A working group will provide expertise on the cybersecurity of platform IT systems across DOD. The new guidelines will be rolled out across the U.S. and international defense partners. The guidelines are required to be submitted by agencies to comply with the requirements of compliance. Defense contractors must demonstrate adherence to these standards through regular audits and assessments. The implementation timeline varies by agency size and existing infrastructure capabilities."""

with gr.Blocks(title="FactShield Demo") as demo:
    # Custom CSS to center content
    gr.HTML("""
    <style>
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .main {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    </style>
    """)
    
    gr.Markdown("# FactShield — Evidence-Grounded Summarization")
    gr.Markdown("""
    **Instructions:**
    - Upload a text file (.txt) OR paste a long document (1000+ words recommended)
    - The system will generate a summary with citations and factuality verification
    - **Tip**: Longer documents (2-5 pages) produce better results with more citations
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Document (Optional)",
                file_types=[".txt"],
                type="filepath"
            )
            inp = gr.Textbox(
                lines=20, 
                label="Input Document (or paste text here)",
                placeholder="Paste your document here (1000+ words recommended for best results)...",
                value=EXAMPLE_TEXT
            )
            gr.Examples(
                examples=[[EXAMPLE_TEXT]],
                inputs=inp,
                label="Example Document"
            )
            btn = gr.Button("Summarize with Citations + Verify", variant="primary")
        
        with gr.Column(scale=1):
            out_summary = gr.Markdown(label="Summary with Citations")
            out_score = gr.Markdown(label="Factuality Score")
            out_detail = gr.Markdown(label="Sentence Verification Details")
    
    def handle_file_or_text(file, text):
        if file:
            file_text = process_file(file)
            if file_text and not file_text.startswith("Error"):
                return file_text
        return text
    
    file_input.change(
        fn=handle_file_or_text,
        inputs=[file_input, inp],
        outputs=inp
    )
    
    btn.click(
        summarize, 
        inputs=inp, 
        outputs=[out_summary, out_score, out_detail]
    )

if __name__ == '__main__':
    demo.launch(share=True, server_name="0.0.0.0")

import argparse, os, glob, hashlib
from tqdm import tqdm
from transformers import AutoTokenizer
from src.utils import write_jsonl

# CPU‑friendly chunker: token windows with overlap

def chunk_text(text, tokenizer, chunk_tokens=900, stride=200):
    """
    Split text into overlapping chunks using tokenizer.
    
    Args:
        text: Input text to chunk
        tokenizer: HuggingFace tokenizer instance
        chunk_tokens: Maximum tokens per chunk (default: 900)
        stride: Overlap size in tokens (default: 200)
    
    Returns:
        List of tuples: (start_token, end_token, chunk_text)
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        >>> chunks = chunk_text("Long text here...", tok, chunk_tokens=500, stride=100)
        >>> len(chunks)  # Number of chunks
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(tokens), chunk_tokens - stride):
        end = min(start + chunk_tokens, len(tokens))
        piece = tokenizer.decode(tokens[start:end])
        chunks.append((start, end, piece))
        if end == len(tokens):
            break
    return chunks

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokenizer', default='sshleifer/distilbart-cnn-12-6')
    ap.add_argument('--chunk_tokens', type=int, default=900)
    ap.add_argument('--stride', type=int, default=200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    files = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    rows = []
    for fp in tqdm(files):
        if not os.path.isfile(fp):
            continue
        doc_id = os.path.splitext(os.path.basename(fp))[0]
        with open(fp, 'r', encoding='utf-8') as f:
            text = f.read()
        for i, (s, e, piece) in enumerate(chunk_text(text, tok, args.chunk_tokens, args.stride)):
            rows.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_{i:04d}",
                'start_token': s,
                'end_token': e,
                'text': piece
            })
    write_jsonl(args.out, rows)
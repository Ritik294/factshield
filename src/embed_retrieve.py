import argparse, json, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from src.utils import read_jsonl, write_jsonl


# Build embeddings


def build_embeddings(chunks_path, emb_out, meta_out, model_name='all-MiniLM-L6-v2', batch=128):
model = SentenceTransformer(model_name, device='cpu')
texts, meta = [], []
for r in read_jsonl(chunks_path):
texts.append(r['text'])
meta.append({'doc_id': r['doc_id'], 'chunk_id': r['chunk_id']})
embs = model.encode(texts, convert_to_numpy=True, batch_size=batch, show_progress_bar=True, normalize_embeddings=True)
np.save(emb_out, embs)
with open(meta_out, 'w', encoding='utf-8') as f:
json.dump(meta, f)


# Build FAISS index


def build_index(emb_path, index_out):
embs = np.load(emb_path)
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs)
faiss.write_index(index, index_out)


# Retrieve top‑k per doc


def retrieve_topk(chunks_path, meta_path, index_path, top_k, out_path):
index = faiss.read_index(index_path)
with open(meta_path, 'r', encoding='utf-8') as f:
meta = json.load(f)
# Query by each doc's first chunk centroid (simple & fast)
by_doc = {}
for r in read_jsonl(chunks_path):
by_doc.setdefault(r['doc_id'], []).append(r)
rows = []
for doc_id, chunks in tqdm(by_doc.items()):
# pick middle chunk as query to avoid intro bias
mid = len(chunks)//2
q = chunks[mid]['text']
qemb = SentenceTransformer('all-MiniLM-L6-v2', device='cpu').encode([q], convert_to_numpy=True, normalize_embeddings=True)
D, I = index.search(qemb, top_k)
top = [meta[i] for i in I[0]]
rows.append({'doc_id': doc_id, 'topk': top})
write_jsonl(out_path, rows)


if __name__ == '__main__':
ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest='cmd')


b1 = sub.add_parser('build-emb')
b1.add_argument('--chunks', required=True)
b1.add_argument('--emb_out', required=True)
b1.add_argument('--meta_out', required=True)


b2 = sub.add_parser('build-index')
b2.add_argument('--emb', required=True)
b2.add_argument('--index_out', required=True)


b3 = sub.add_parser('retrieve')
b3.add_argument('--chunks', required=True)
b3.add_argument('--meta', required=True)
b3.add_argument('--index', required=True)
b3.add_argument('--top_k', type=int, default=3)
b3.add_argument('--out', required=True)


args = ap.parse_args()
if args.cmd == 'build-emb':
build_embeddings(args.chunks, args.emb_out, args.meta_out)
elif args.cmd == 'build-index':
build_index(args.emb, args.index_out)
elif args.cmd == 'retrieve':
retrieve_topk(args.chunks, args.meta, args.index, args.top_k, args.out)
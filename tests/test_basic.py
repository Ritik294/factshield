import os, json
from src.chunker import chunk_text
from transformers import AutoTokenizer


def test_chunker_small():
tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
text = 'A '*5000
chunks = chunk_text(text, tok, chunk_tokens=200, stride=50)
assert len(chunks) >= 5
# continuity
assert chunks[0][0] == 0


def test_citation_regex():
from src.cite import extract_citations
s = 'Numbers rose [CIT:doc:chunk0001] and fell [CIT:doc:chunk0002].'
c = extract_citations(s)
assert len(c) == 2
assert c[0]['chunk'].startswith('chunk')
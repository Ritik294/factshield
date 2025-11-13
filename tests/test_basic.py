import os, json
from src.chunker import chunk_text
from transformers import AutoTokenizer

def test_chunker_small():
    tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    text = 'A '*5000
    chunks = chunk_text(text, tok, chunk_tokens=200, stride=50)
    assert len(chunks) >= 5
    assert chunks[0][0] == 0

def test_citation_regex():
    from src.cite import extract_citations
    s = 'Numbers rose [CIT:doc:chunk0001] and fell [CIT:doc:chunk0002].'
    c = extract_citations(s)
    assert len(c) == 2
    assert c[0]['chunk'].startswith('chunk')

def test_citation_edge_cases():
    """Test citation extraction edge cases."""
    from src.cite import extract_citations
    
    # Citation at start
    s1 = "[CIT:doc:chunk1] This is a sentence."
    assert len(extract_citations(s1)) == 1
    
    # Citation at end
    s2 = "This is a sentence [CIT:doc:chunk1]."
    assert len(extract_citations(s2)) == 1
    
    # Multiple citations
    s3 = "First [CIT:doc:chunk1] and second [CIT:doc:chunk2]."
    assert len(extract_citations(s3)) == 2
    
    # No citations
    s4 = "This has no citations."
    assert len(extract_citations(s4)) == 0

def test_chunking_boundaries():
    """Test chunking handles boundaries correctly."""
    tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    
    # Very short text
    short = "Short text."
    chunks = chunk_text(short, tok, chunk_tokens=900, stride=200)
    assert len(chunks) >= 1
    
    # Exact boundary
    exact = 'Word ' * 900
    chunks = chunk_text(exact, tok, chunk_tokens=900, stride=200)
    assert len(chunks) >= 1
    
    # Overlap check
    long_text = 'Word ' * 2000
    chunks = chunk_text(long_text, tok, chunk_tokens=900, stride=200)
    assert len(chunks) >= 2
    # Check overlap
    assert chunks[1][0] < chunks[0][1]  # Next chunk starts before previous ends
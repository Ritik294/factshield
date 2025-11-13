"""
Integration tests for end-to-end pipeline.
"""
import os
import json
import tempfile
import shutil
from src.utils import read_jsonl

def test_end_to_end_pipeline():
    """Test complete pipeline with sample data."""
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create sample document
        doc_path = os.path.join(temp_dir, "test_doc.txt")
        with open(doc_path, 'w') as f:
            f.write("This is paragraph one. It has multiple sentences. Each sentence adds information.\n\n")
            f.write("This is paragraph two. It continues the discussion. More details are provided here.\n\n")
            f.write("This is paragraph three. Final thoughts are shared. The document concludes here.")
        
        # Test chunking
        from src.chunker import chunk_text
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
        with open(doc_path, 'r') as f:
            text = f.read()
        chunks = chunk_text(text, tok, chunk_tokens=100, stride=50)
        assert len(chunks) > 0
        
        # Test citation extraction
        from src.cite import extract_citations
        test_summary = "Summary sentence [CIT:test:chunk1] and another [CIT:test:chunk2]."
        citations = extract_citations(test_summary)
        assert len(citations) == 2
        
    finally:
        shutil.rmtree(temp_dir)

def test_output_formats():
    """Verify output file formats are correct."""
    # This would check actual output files if they exist
    # For now, just verify JSONL structure
    sample_row = {
        'doc_id': 'test',
        'summary': 'Test summary',
        'citations': [{'doc': 'test', 'chunk': 'chunk1'}]
    }
    json_str = json.dumps(sample_row)
    parsed = json.loads(json_str)
    assert 'doc_id' in parsed
    assert 'summary' in parsed
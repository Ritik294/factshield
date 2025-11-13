"""
Utility functions for reading and writing JSONL files.
"""
import json, os
from typing import Iterable, Dict, Any

def read_jsonl(path: str):
    """
    Read JSONL file line by line.
    
    Args:
        path: Path to JSONL file
    
    Yields:
        Dictionary parsed from each line
    
    Example:
        >>> for row in read_jsonl('data.jsonl'):
        ...     print(row['doc_id'])
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    """
    Write rows to JSONL file.
    
    Args:
        path: Output file path (directory will be created if needed)
        rows: Iterable of dictionaries to write
    
    Example:
        >>> rows = [{'doc_id': '1', 'text': 'Hello'}]
        >>> write_jsonl('output.jsonl', rows)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
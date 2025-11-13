"""
Citation extraction utilities.
"""
import re
from typing import List, Dict

CIT_RE = re.compile(r"\[CIT:(?P<doc>[^:\]]+):(?P<chunk>[^\]]+)\]")

def extract_citations(summary: str) -> List[Dict[str, str]]:
    """
    Extract citation tags from a summary text.
    
    Args:
        summary: Text containing citations in format [CIT:doc_id:chunk_id]
    
    Returns:
        List of dictionaries with 'doc' and 'chunk' keys
    
    Example:
        >>> extract_citations("Text [CIT:doc1:chunk1] more [CIT:doc1:chunk2]")
        [{'doc': 'doc1', 'chunk': 'chunk1'}, {'doc': 'doc1', 'chunk': 'chunk2'}]
        
        >>> extract_citations("No citations here")
        []
    """
    return [m.groupdict() for m in CIT_RE.finditer(summary)]
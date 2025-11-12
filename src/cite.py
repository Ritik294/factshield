import re
from typing import List, Dict

CIT_RE = re.compile(r"\[CIT:(?P<doc>[^:\]]+):(?P<chunk>[^\]]+)\]")

# Extract unique citation tags in a summary

def extract_citations(summary: str) -> List[Dict[str,str]]:
    return [m.groupdict() for m in CIT_RE.finditer(summary)]
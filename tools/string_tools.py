import re

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)
#!/usr/bin/env python3
from __future__ import annotations
import re
_WS = re.compile(r"\s+")
def clean_text(s: str) -> str:
    if not isinstance(s, str): return ''
    s = s.lower()
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r"[^a-z0-9\s']", ' ', s)
    return _WS.sub(' ', s).strip()

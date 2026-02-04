from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

@dataclass(frozen=True)
class DocPage:
    source: str
    page: int
    text: str

def load_files(paths: Iterable[Path]) -> List[DocPage]:
    pages: List[DocPage] = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            reader = PdfReader(str(p))
            for i, page in enumerate(reader.pages, start=1):
                txt = (page.extract_text() or "").strip()
                if txt:
                    pages.append(DocPage(source=p.name, page=i, text=txt))
        elif p.suffix.lower() == ".txt":
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                pages.append(DocPage(source=p.name, page=1, text=txt))
    return pages

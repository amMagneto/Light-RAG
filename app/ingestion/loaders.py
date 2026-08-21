from dataclasses import dataclass, field
from pathlib import Path

import docx2txt
from pypdf import PdfReader

from app.core.exceptions import UnsupportedFormatError

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class ParsedPage:
    page: int | None
    text: str


@dataclass
class ParsedDocument:
    path: Path
    mime_type: str
    pages: list[ParsedPage] = field(default_factory=list)
    page_count: int | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def size_kb(self) -> float:
        return round(self.path.stat().st_size / 1024, 2) if self.path.exists() else 0.0


def parse(path_str: str) -> ParsedDocument:
    path = Path(path_str)
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages = [
            ParsedPage(page=i + 1, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        ]
        return ParsedDocument(path=path, mime_type="application/pdf", pages=pages, page_count=len(pages))
    if ext == ".docx":
        text = docx2txt.process(str(path)) or ""
        return ParsedDocument(
            path=path,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            pages=[ParsedPage(page=None, text=text)],
            page_count=None,
        )
    if ext == ".txt":
        return ParsedDocument(
            path=path,
            mime_type="text/plain",
            pages=[ParsedPage(page=None, text=path.read_text(encoding="utf-8", errors="ignore"))],
            page_count=None,
        )
    if ext == ".md":
        return ParsedDocument(
            path=path,
            mime_type="text/markdown",
            pages=[ParsedPage(page=None, text=path.read_text(encoding="utf-8", errors="ignore"))],
            page_count=None,
        )
    raise UnsupportedFormatError(f"Unsupported document format: {ext}")

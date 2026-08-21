import hashlib
import re
from dataclasses import dataclass, field

from app.ingestion.loaders import ParsedDocument


@dataclass
class Section:
    path: list[str]
    blocks: list[tuple[str, int | None]] = field(default_factory=list)


@dataclass
class ChunkData:
    content: str
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = field(default_factory=list)
    token_count: int = 0
    content_hash: str = ""

    def finalize(self) -> "ChunkData":
        self.token_count = max(1, len(self.content) // 4)
        self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        return self


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[\.\)]\s+([A-Za-z].*)$")
_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 ,&/\-()']{2,99}$")


def _heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 100 or stripped.endswith((".", ",", ";", ":", "?", "!")):
        return None
    match = _MARKDOWN_HEADING.match(stripped)
    if match:
        return len(match.group(1)), match.group(2).strip()
    match = _NUMBERED_HEADING.match(stripped)
    if match and len(stripped) <= 100:
        depth = match.group(1).count(".") + 1
        return min(depth + 1, 6), f"{match.group(1)} {match.group(2)}".strip()
    words = stripped.split()
    if _ALLCAPS.match(stripped) and 1 <= len(words) <= 12:
        return 1, stripped
    if 2 <= len(words) <= 9:
        capitalized = sum(1 for word in words if word[:1].isupper())
        if capitalized >= len(words) - (1 if len(words) > 3 else 0) and not stripped.endswith("."):
            return 3, stripped
    return None


def build_sections(parsed: ParsedDocument) -> list[Section]:
    sections: list[Section] = []
    stack: list[str] = []
    current = Section(path=[])
    sections.append(current)

    for page in parsed.pages:
        buffer: list[str] = []
        buffer_page: int | None = page.page

        def flush() -> None:
            nonlocal buffer
            if buffer:
                text = "\n".join(buffer).strip()
                if text:
                    current.blocks.append((text, buffer_page))
                buffer = []

        for line in page.text.splitlines():
            heading = _heading(line)
            if heading is None:
                if line.strip():
                    buffer.append(line.rstrip())
                else:
                    flush()
                continue
            flush()
            level, title = heading
            stack = stack[: level - 1]
            stack.append(title)
            current = Section(path=list(stack))
            sections.append(current)
        flush()

    return [section for section in sections if section.blocks]


def choose_chunk_config(page_count: int | None, file_size_kb: float) -> tuple[int, int]:
    if page_count is not None:
        if page_count <= 2:
            return 300, 50
        if page_count <= 15:
            return 700, 150
        if page_count <= 100:
            return 1200, 250
        return 2000, 400
    if file_size_kb < 50:
        return 300, 50
    if file_size_kb < 500:
        return 700, 150
    if file_size_kb < 5000:
        return 1200, 250
    return 2000, 400


def _tail(text: str, size: int) -> str:
    if len(text) <= size:
        return text
    cut = text[-size:]
    space = cut.find(" ")
    return cut[space + 1 :] if space != -1 else cut


def pack_sections(
    sections: list[Section],
    chunk_size: int,
    overlap: int,
) -> list[ChunkData]:
    chunks: list[ChunkData] = []
    index = 0

    for section in sections:
        current_parts: list[tuple[str, int | None]] = []
        current_len = 0
        pages_seen: list[int] = []

        def emit() -> None:
            nonlocal current_parts, current_len, pages_seen, index
            if not current_parts:
                return
            content = "\n\n".join(part for part, _ in current_parts).strip()
            if content:
                clean_pages = sorted(set(p for p in pages_seen if p is not None))
                chunks.append(
                    ChunkData(
                        content=content,
                        chunk_index=index,
                        page_start=clean_pages[0] if clean_pages else None,
                        page_end=clean_pages[-1] if clean_pages else None,
                        section_path=list(section.path),
                    ).finalize()
                )
                index += 1
            current_parts = []
            current_len = 0
            pages_seen = []

        for block_text, block_page in section.blocks:
            parts = [block_text]
            while parts:
                part = parts[0]
                part_len = len(part) + 2

                if part_len > chunk_size:
                    emit()
                    hard_pieces = [
                        piece for piece in _hard_split(part, chunk_size - overlap)
                        if piece.strip()
                    ]
                    last_piece = None
                    for piece in hard_pieces:
                        pages = [block_page] if block_page is not None else []
                        chunks.append(
                            ChunkData(
                                content=piece.strip(),
                                chunk_index=index,
                                page_start=block_page,
                                page_end=block_page,
                                section_path=list(section.path),
                            ).finalize()
                        )
                        index += 1
                        last_piece = piece
                    if last_piece is not None and overlap > 0:
                        carry = _tail(last_piece, overlap)
                        if carry.strip():
                            current_parts = [(carry, block_page)]
                            current_len = len(carry)
                            pages_seen = [block_page] if block_page is not None else []
                    parts.pop(0)
                    continue

                if current_len + part_len > chunk_size and current_parts:
                    emit()
                    if overlap > 0:
                        carry_source = "\n\n".join(part_ for part_, _ in current_parts) if current_parts else ""
                        carry = _tail(carry_source, overlap)
                        if carry.strip():
                            current_parts = [(carry, block_page)]
                            current_len = len(carry)
                            pages_seen = [block_page] if block_page is not None else []

                current_parts.append((part, block_page))
                current_len += part_len
                if block_page is not None:
                    pages_seen.append(block_page)
                parts.pop(0)

        emit()

    return chunks


def _hard_split(text: str, window: int) -> list[str]:
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        if end < len(text):
            period = text.rfind(". ", start, end)
            if period > start + window // 2:
                end = period + 1
        pieces.append(text[start:end])
        start = end
    return pieces


def chunk_document(parsed: ParsedDocument, default_size: int, default_overlap: int) -> list[ChunkData]:
    sections = build_sections(parsed)
    if not sections:
        return []
    page_count = parsed.page_count
    size, overlap = choose_chunk_config(page_count, parsed.size_kb) if page_count else (
        default_size,
        default_overlap,
    )
    return pack_sections(sections, size, overlap)

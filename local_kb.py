"""
Lightweight local knowledge base with paragraph-level evidence定位.

Design goals:
- No mandatory third-party dependencies.
- SQLite metadata / paragraph / chunk store.
- Offline keyword/BM25-style retrieval.
- Paragraph-level evidence metadata for local documents.
- Optional PDF extraction when pypdf or PyPDF2 is already installed.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

SUPPORTED_EXTS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp", ".h", ".go", ".rs",
    ".html", ".htm",
    ".json", ".jsonl", ".csv",
    ".ipynb", ".docx", ".pdf",
}

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 160
MAX_TEXT_CHARS_PER_DOC = 1_200_000


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = (tag or "").lower()
        if tag in {"script", "style", "noscript", "svg", "canvas", "iframe"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "blockquote"}:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        tag = (tag or "").lower()
        if tag in {"script", "style", "noscript", "svg", "canvas", "iframe"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "blockquote"}:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip_depth and data:
            self.parts.append(data)

    def get_text(self) -> str:
        return clean_text("".join(self.parts))


@dataclass
class KBPaths:
    base_dir: Path
    kb_dir: Path
    docs_dir: Path
    index_dir: Path
    db_path: Path


def make_paths(base_dir: str | Path, docs_dir: str | Path | None = None, db_path: str | Path | None = None) -> KBPaths:
    base = Path(base_dir).resolve()
    kb_dir = base / "kb"
    docs = Path(docs_dir).resolve() if docs_dir else kb_dir / "documents"
    index = kb_dir / "index"
    db = Path(db_path).resolve() if db_path else kb_dir / "kb.sqlite"
    return KBPaths(base, kb_dir, docs, index, db)


def ensure_kb_layout(base_dir: str | Path, docs_dir: str | Path | None = None, db_path: str | Path | None = None) -> KBPaths:
    paths = make_paths(base_dir, docs_dir, db_path)
    paths.kb_dir.mkdir(parents=True, exist_ok=True)
    paths.docs_dir.mkdir(parents=True, exist_ok=True)
    paths.index_dir.mkdir(parents=True, exist_ok=True)
    init_db(paths.db_path)
    return paths


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if column not in _column_names(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_db(db_path: str | Path):
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    with connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                rel_path TEXT NOT NULL,
                title TEXT NOT NULL,
                ext TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paragraphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                paragraph_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                char_start INTEGER NOT NULL DEFAULT 0,
                char_end INTEGER NOT NULL DEFAULT 0,
                page_number INTEGER,
                FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                terms TEXT NOT NULL,
                paragraph_start INTEGER,
                paragraph_end INTEGER,
                char_start INTEGER NOT NULL DEFAULT 0,
                char_end INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
            CREATE INDEX IF NOT EXISTS idx_paragraphs_doc_id ON paragraphs(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            """
        )
        # Old kb.sqlite may already have chunks without these columns.
        _add_column_if_missing(conn, "chunks", "paragraph_start", "paragraph_start INTEGER")
        _add_column_if_missing(conn, "chunks", "paragraph_end", "paragraph_end INTEGER")
        _add_column_if_missing(conn, "chunks", "char_start", "char_start INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "chunks", "char_end", "char_end INTEGER NOT NULL DEFAULT 0")
        conn.commit()


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\x00", " ")
    # Keep paragraph breaks. Collapse spaces inside lines.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""


def read_html(path: Path) -> str:
    raw = safe_read_text(path)
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(raw)
        return extractor.get_text()
    except Exception:
        return clean_text(re.sub(r"<[^>]+>", "\n\n", raw))


def read_csv(path: Path) -> str:
    raw = safe_read_text(path)
    if not raw:
        return ""
    lines = []
    try:
        sample = raw[:4096]
        dialect = csv.Sniffer().sniff(sample)
    except Exception:
        dialect = csv.excel
    try:
        reader = csv.reader(raw.splitlines(), dialect)
        for i, row in enumerate(reader):
            line = " | ".join(cell.strip() for cell in row if cell.strip())
            if line:
                lines.append(line)
            if i >= 8000:
                break
    except Exception:
        return raw
    return clean_text("\n\n".join(lines))


def read_json(path: Path) -> str:
    raw = safe_read_text(path)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        return clean_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        return raw


def read_ipynb(path: Path) -> str:
    raw = safe_read_text(path)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        parts = []
        for cell in data.get("cells", []):
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else str(source)
            if text.strip():
                parts.append(text)
        return clean_text("\n\n".join(parts))
    except Exception:
        return raw


def read_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            names = ["word/document.xml"]
            names += [name for name in zf.namelist() if name.startswith("word/header") or name.startswith("word/footer")]
            names += [name for name in zf.namelist() if name.startswith("word/footnotes") or name.startswith("word/endnotes")]
            parts = []
            for name in names:
                if name not in zf.namelist():
                    continue
                xml_text = zf.read(name).decode("utf-8", errors="ignore")
                # Preserve Word paragraphs.
                xml_text = re.sub(r"</w:p>", "\n\n", xml_text)
                xml_text = re.sub(r"<w:tab\s*/>", "\t", xml_text)
                xml_text = re.sub(r"<w:br\s*/>", "\n", xml_text)
                xml_text = re.sub(r"<[^>]+>", " ", xml_text)
                if xml_text.strip():
                    parts.append(xml_text)
            return clean_text("\n\n".join(parts))
    except Exception:
        return ""


def read_pdf_optional(path: Path) -> str:
    try:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return ""

    try:
        reader = PdfReader(str(path))
        parts = []
        for page_index, page in enumerate(reader.pages[:300], start=1):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"[PDF 第 {page_index} 页]\n" + text)
        return clean_text("\n\n".join(parts))
    except Exception:
        return ""


def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".html", ".htm"}:
        text = read_html(path)
    elif ext == ".csv":
        text = read_csv(path)
    elif ext in {".json", ".jsonl"}:
        text = read_json(path)
    elif ext == ".ipynb":
        text = read_ipynb(path)
    elif ext == ".docx":
        text = read_docx(path)
    elif ext == ".pdf":
        text = read_pdf_optional(path)
    else:
        text = safe_read_text(path)

    text = clean_text(text)
    if len(text) > MAX_TEXT_CHARS_PER_DOC:
        text = text[:MAX_TEXT_CHARS_PER_DOC]
    return text


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def iter_documents(docs_dir: Path) -> Iterable[Path]:
    if not docs_dir.exists():
        return []
    for path in docs_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def split_paragraphs(text: str) -> list[dict]:
    text = clean_text(text)
    if not text:
        return []

    raw_parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if len(raw_parts) <= 1:
        # Fallback: line-level paragraphs for DOCX/plain text that only has single newlines.
        raw_parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    if len(raw_parts) <= 1 and len(text) > 900:
        raw_parts = [p.strip() for p in re.split(r"(?<=[。！？!?\.])\s+", text) if p.strip()]

    paragraphs = []
    cursor = 0
    for part in raw_parts:
        part = clean_text(part)
        if not part:
            continue
        found = text.find(part, cursor)
        if found < 0:
            found = cursor
        end = found + len(part)
        paragraphs.append({
            "paragraph_index": len(paragraphs) + 1,
            "text": part,
            "char_start": found,
            "char_end": end,
            "page_number": _detect_pdf_page_number(part),
        })
        cursor = end
    return paragraphs


def _detect_pdf_page_number(text: str) -> int | None:
    m = re.match(r"\[PDF 第 (\d+) 页\]", text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def build_chunks_from_paragraphs(
    paragraphs: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    chunks: list[dict] = []
    current_text = ""
    start_para = None
    end_para = None
    char_start = 0
    char_end = 0

    def flush_current() -> None:
        nonlocal current_text, start_para, end_para, char_start, char_end
        text = clean_text(current_text)
        if len(text) >= 40 and start_para is not None and end_para is not None:
            chunks.append({
                "chunk_index": len(chunks),
                "text": text,
                "paragraph_start": int(start_para),
                "paragraph_end": int(end_para),
                "char_start": int(char_start or 0),
                "char_end": int(char_end or 0),
            })
        current_text = ""
        start_para = None
        end_para = None
        char_start = 0
        char_end = 0

    for paragraph in paragraphs:
        p_text = paragraph["text"]
        p_idx = int(paragraph["paragraph_index"])

        if len(p_text) > chunk_size * 1.5:
            flush_current()
            step = max(1, chunk_size - overlap)
            for offset in range(0, len(p_text), step):
                part = p_text[offset:offset + chunk_size].strip()
                if len(part) >= 80:
                    chunks.append({
                        "chunk_index": len(chunks),
                        "text": part,
                        "paragraph_start": p_idx,
                        "paragraph_end": p_idx,
                        "char_start": int(paragraph["char_start"] + offset),
                        "char_end": int(paragraph["char_start"] + offset + len(part)),
                    })
            continue

        if not current_text:
            current_text = p_text
            start_para = p_idx
            end_para = p_idx
            char_start = int(paragraph["char_start"])
            char_end = int(paragraph["char_end"])
            continue

        if len(current_text) + len(p_text) + 2 <= chunk_size:
            current_text = (current_text + "\n\n" + p_text).strip()
            end_para = p_idx
            char_end = int(paragraph["char_end"])
        else:
            flush_current()
            current_text = p_text
            start_para = p_idx
            end_para = p_idx
            char_start = int(paragraph["char_start"])
            char_end = int(paragraph["char_end"])

    flush_current()
    return chunks


def split_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    paragraphs = split_paragraphs(text)
    return [chunk["text"] for chunk in build_chunks_from_paragraphs(paragraphs, chunk_size=chunk_size, overlap=overlap)]


def extract_terms(text: str) -> list[str]:
    text = (text or "").lower()
    terms: list[str] = []

    for token in re.findall(r"[a-z0-9_][a-z0-9_\-.:/#]{1,}", text):
        token = token.strip(".-:/#")
        if len(token) >= 2:
            terms.append(token)

    for run in re.findall(r"[\u4e00-\u9fff]{1,}", text):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(list(run))
            terms.extend(run[i:i + 2] for i in range(max(0, len(run) - 1)))
            if len(run) >= 3:
                terms.extend(run[i:i + 3] for i in range(max(0, len(run) - 2)))

    seen = set()
    out = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            out.append(term)
    return out


def terms_to_text(text: str) -> str:
    return " ".join(extract_terms(text)[:900])


def _insert_document_index(conn: sqlite3.Connection, path: Path, paths: KBPaths, text: str, mtime: float, size: int, h: str) -> tuple[int, int]:
    paragraphs = split_paragraphs(text)
    chunks = build_chunks_from_paragraphs(paragraphs)
    if not chunks:
        return 0, 0

    rel_path = str(path.relative_to(paths.docs_dir))
    cur = conn.execute(
        """
        INSERT INTO documents(path, rel_path, title, ext, mtime, size, content_hash, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (str(path.resolve()), rel_path, path.stem, path.suffix.lower(), mtime, size, h, time.time()),
    )
    doc_id = int(cur.lastrowid)

    for paragraph in paragraphs:
        conn.execute(
            """
            INSERT INTO paragraphs(doc_id, paragraph_index, text, char_start, char_end, page_number)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                int(paragraph["paragraph_index"]),
                paragraph["text"],
                int(paragraph["char_start"]),
                int(paragraph["char_end"]),
                paragraph.get("page_number"),
            ),
        )

    for idx, chunk in enumerate(chunks):
        conn.execute(
            """
            INSERT INTO chunks(doc_id, chunk_index, text, terms, paragraph_start, paragraph_end, char_start, char_end)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                idx,
                chunk["text"],
                terms_to_text(chunk["text"]),
                chunk.get("paragraph_start"),
                chunk.get("paragraph_end"),
                int(chunk.get("char_start") or 0),
                int(chunk.get("char_end") or 0),
            ),
        )

    return 1, len(chunks)


def rebuild_kb_index(
    base_dir: str | Path,
    docs_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    paths = ensure_kb_layout(base_dir, docs_dir, db_path)
    started = time.time()
    stats = {
        "ok": True,
        "mode": "rebuild",
        "documents_dir": str(paths.docs_dir),
        "db_path": str(paths.db_path),
        "doc_count": 0,
        "chunk_count": 0,
        "paragraph_count": 0,
        "skipped_count": 0,
        "errors": [],
    }

    with connect(paths.db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM paragraphs")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")

        for path in iter_documents(paths.docs_dir):
            try:
                text = read_document(path)
                if not text or len(text) < 20:
                    stats["skipped_count"] += 1
                    if path.suffix.lower() == ".pdf":
                        stats["errors"].append(f"跳过 PDF：{path.name}（未安装 pypdf/PyPDF2 或无法提取文本）")
                    continue

                paragraphs = split_paragraphs(text)
                chunks = build_chunks_from_paragraphs(paragraphs, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    stats["skipped_count"] += 1
                    continue

                st = path.stat()
                rel_path = str(path.relative_to(paths.docs_dir))
                h = content_hash(text)
                cur = conn.execute(
                    """
                    INSERT INTO documents(path, rel_path, title, ext, mtime, size, content_hash, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (str(path.resolve()), rel_path, path.stem, path.suffix.lower(), st.st_mtime, st.st_size, h, time.time()),
                )
                doc_id = int(cur.lastrowid)

                for paragraph in paragraphs:
                    conn.execute(
                        """
                        INSERT INTO paragraphs(doc_id, paragraph_index, text, char_start, char_end, page_number)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            int(paragraph["paragraph_index"]),
                            paragraph["text"],
                            int(paragraph["char_start"]),
                            int(paragraph["char_end"]),
                            paragraph.get("page_number"),
                        ),
                    )

                for idx, chunk in enumerate(chunks):
                    conn.execute(
                        """
                        INSERT INTO chunks(doc_id, chunk_index, text, terms, paragraph_start, paragraph_end, char_start, char_end)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            idx,
                            chunk["text"],
                            terms_to_text(chunk["text"]),
                            chunk.get("paragraph_start"),
                            chunk.get("paragraph_end"),
                            int(chunk.get("char_start") or 0),
                            int(chunk.get("char_end") or 0),
                        ),
                    )

                stats["doc_count"] += 1
                stats["paragraph_count"] += len(paragraphs)
                stats["chunk_count"] += len(chunks)
            except Exception as exc:
                stats["errors"].append(f"{path.name}: {str(exc)[:160]}")

        conn.commit()

    stats["seconds"] = round(time.time() - started, 2)
    return stats


def sync_kb_index(
    base_dir: str | Path,
    docs_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    paths = ensure_kb_layout(base_dir, docs_dir, db_path)
    started = time.time()
    stats = {
        "ok": True,
        "mode": "sync",
        "documents_dir": str(paths.docs_dir),
        "db_path": str(paths.db_path),
        "added_count": 0,
        "updated_count": 0,
        "removed_count": 0,
        "unchanged_count": 0,
        "skipped_count": 0,
        "chunk_count": 0,
        "errors": [],
    }

    current_files = {}
    for path in iter_documents(paths.docs_dir):
        try:
            current_files[str(path.resolve())] = path
        except Exception:
            continue

    with connect(paths.db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        existing_rows = conn.execute("SELECT id, path, mtime, size, content_hash FROM documents").fetchall()
        existing = {row["path"]: row for row in existing_rows}

        for indexed_path, row in list(existing.items()):
            if indexed_path not in current_files:
                conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
                stats["removed_count"] += 1
                existing.pop(indexed_path, None)

        for resolved_path, path in current_files.items():
            try:
                st = path.stat()
                row = existing.get(resolved_path)
                if row and float(row["mtime"]) == float(st.st_mtime) and int(row["size"]) == int(st.st_size):
                    # If old DB has no paragraph metadata for this doc, force rebuild the doc once.
                    para_count = conn.execute("SELECT COUNT(*) FROM paragraphs WHERE doc_id = ?", (row["id"],)).fetchone()[0]
                    if para_count:
                        stats["unchanged_count"] += 1
                        continue

                text = read_document(path)
                if not text or len(text) < 20:
                    if row:
                        conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
                        stats["removed_count"] += 1
                    stats["skipped_count"] += 1
                    if path.suffix.lower() == ".pdf":
                        stats["errors"].append(f"跳过 PDF：{path.name}（未安装 pypdf/PyPDF2 或无法提取文本）")
                    continue

                h = content_hash(text)
                if row and row["content_hash"] == h:
                    # Content same, but paragraph table missing/old schema may need rebuild.
                    para_count = conn.execute("SELECT COUNT(*) FROM paragraphs WHERE doc_id = ?", (row["id"],)).fetchone()[0]
                    chunk_cols = _column_names(conn, "chunks")
                    has_chunk_location = {"paragraph_start", "paragraph_end"}.issubset(chunk_cols)
                    if para_count and has_chunk_location:
                        conn.execute(
                            "UPDATE documents SET mtime = ?, size = ?, indexed_at = ? WHERE id = ?",
                            (st.st_mtime, st.st_size, time.time(), row["id"]),
                        )
                        stats["unchanged_count"] += 1
                        continue

                paragraphs = split_paragraphs(text)
                chunks = build_chunks_from_paragraphs(paragraphs, chunk_size=chunk_size, overlap=overlap)
                if not chunks:
                    if row:
                        conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
                        stats["removed_count"] += 1
                    stats["skipped_count"] += 1
                    continue

                if row:
                    conn.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
                    stats["updated_count"] += 1
                else:
                    stats["added_count"] += 1

                rel_path = str(path.relative_to(paths.docs_dir))
                cur = conn.execute(
                    """
                    INSERT INTO documents(path, rel_path, title, ext, mtime, size, content_hash, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (resolved_path, rel_path, path.stem, path.suffix.lower(), st.st_mtime, st.st_size, h, time.time()),
                )
                doc_id = int(cur.lastrowid)

                for paragraph in paragraphs:
                    conn.execute(
                        """
                        INSERT INTO paragraphs(doc_id, paragraph_index, text, char_start, char_end, page_number)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            int(paragraph["paragraph_index"]),
                            paragraph["text"],
                            int(paragraph["char_start"]),
                            int(paragraph["char_end"]),
                            paragraph.get("page_number"),
                        ),
                    )

                for idx, chunk in enumerate(chunks):
                    conn.execute(
                        """
                        INSERT INTO chunks(doc_id, chunk_index, text, terms, paragraph_start, paragraph_end, char_start, char_end)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            idx,
                            chunk["text"],
                            terms_to_text(chunk["text"]),
                            chunk.get("paragraph_start"),
                            chunk.get("paragraph_end"),
                            int(chunk.get("char_start") or 0),
                            int(chunk.get("char_end") or 0),
                        ),
                    )
                stats["chunk_count"] += len(chunks)
            except Exception as exc:
                stats["errors"].append(f"{path.name}: {str(exc)[:160]}")

        conn.commit()

    stats["changed"] = bool(stats["added_count"] or stats["updated_count"] or stats["removed_count"] or stats["skipped_count"])
    stats["seconds"] = round(time.time() - started, 2)
    return stats


def _score_chunk(query_terms: list[str], phrase: str, title: str, rel_path: str, text: str, terms_blob: str) -> float:
    if not query_terms:
        return 0.0

    lower_title = (title or "").lower()
    lower_path = (rel_path or "").lower()
    lower_text = (text or "").lower()
    lower_terms = (terms_blob or "").lower()
    lower_phrase = (phrase or "").lower().strip()

    score = 0.0
    matched = 0

    if lower_phrase and len(lower_phrase) >= 3:
        if lower_phrase in lower_text:
            score += 9.0
        if lower_phrase in lower_title or lower_phrase in lower_path:
            score += 8.0

    for term in query_terms:
        term = term.lower()
        if not term:
            continue
        hit = 0
        if term in lower_title:
            hit += 5
        if term in lower_path:
            hit += 3
        if term in lower_terms:
            hit += 2
        count = lower_text.count(term)
        if count:
            hit += min(6, 1 + count)
        if hit:
            matched += 1
            score += hit * (1.0 + min(len(term), 12) / 18.0)

    coverage = matched / max(1, len(query_terms))
    score += coverage * 8.0
    return round(score, 3)


def file_url(path: str | Path) -> str:
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return "file:///" + quote(str(path).replace("\\", "/"))


def _format_location(rel_path: str, ext: str, p_start: int | None, p_end: int | None, page_number: int | None = None) -> str:
    if page_number:
        return f"{rel_path}，PDF 第 {page_number} 页"
    if p_start and p_end:
        if int(p_start) == int(p_end):
            return f"{rel_path}，第 {int(p_start)} 段"
        return f"{rel_path}，第 {int(p_start)}–{int(p_end)} 段"
    return rel_path




def _best_paragraph_evidence(
    conn: sqlite3.Connection,
    doc_id: int,
    paragraph_start: int | None,
    paragraph_end: int | None,
    query_terms: list[str],
    phrase: str,
) -> dict:
    if not paragraph_start or not paragraph_end:
        return {"paragraph_start": paragraph_start, "paragraph_end": paragraph_end, "quote": "", "page_number": None}

    rows = conn.execute(
        """
        SELECT paragraph_index, text, page_number
        FROM paragraphs
        WHERE doc_id = ? AND paragraph_index BETWEEN ? AND ?
        ORDER BY paragraph_index
        """,
        (doc_id, int(paragraph_start), int(paragraph_end)),
    ).fetchall()
    if not rows:
        return {"paragraph_start": paragraph_start, "paragraph_end": paragraph_end, "quote": "", "page_number": None}

    scored = []
    for row in rows:
        s = _score_chunk(query_terms, phrase, "", "", row["text"], terms_to_text(row["text"]))
        scored.append((s, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]

    # 如果相邻下一段很短且包含答案/选项，也一起带上，便于题库证据完整。
    best_idx = int(best["paragraph_index"])
    selected = [best]
    by_idx = {int(row["paragraph_index"]): row for row in rows}
    for near_idx in (best_idx + 1, best_idx + 2, best_idx + 3, best_idx + 4):
        near = by_idx.get(near_idx)
        if not near:
            continue
        near_text = near["text"] or ""
        if re.search(r"^(A|B|C|D|E|F)[\.．、:：]|正确答案|我的答案|答案解析", near_text.strip(), flags=re.I):
            selected.append(near)
        elif len("\n".join(row["text"] for row in selected)) < 420 and len(near_text) < 80:
            selected.append(near)

    selected.sort(key=lambda row: int(row["paragraph_index"]))
    quote = clean_text("\n".join(row["text"] for row in selected))
    if len(quote) > 720:
        quote = quote[:700].rstrip() + "..."
    return {
        "paragraph_start": int(selected[0]["paragraph_index"]),
        "paragraph_end": int(selected[-1]["paragraph_index"]),
        "quote": quote,
        "page_number": best["page_number"],
    }

def search_local_kb(
    query: str,
    base_dir: str | Path,
    docs_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    limit: int = 8,
    min_score: float = 1.0,
) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []

    paths = ensure_kb_layout(base_dir, docs_dir, db_path)
    query_terms = extract_terms(query)[:28]
    if not query_terms:
        return []

    with connect(paths.db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, c.chunk_index, c.text, c.terms,
                   c.paragraph_start, c.paragraph_end, c.char_start, c.char_end,
                   d.id AS doc_id, d.path, d.rel_path, d.title, d.ext, d.mtime
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            """
        ).fetchall()

    scored = []
    for row in rows:
        score = _score_chunk(
            query_terms=query_terms,
            phrase=query,
            title=row["title"],
            rel_path=row["rel_path"],
            text=row["text"],
            terms_blob=row["terms"],
        )
        if score < min_score:
            continue
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen_chunks = set()
    for score, row in scored:
        key = (row["doc_id"], row["chunk_index"])
        if key in seen_chunks:
            continue
        seen_chunks.add(key)

        p_start = row["paragraph_start"]
        p_end = row["paragraph_end"]
        with connect(paths.db_path) as ev_conn:
            ev = _best_paragraph_evidence(
                ev_conn,
                int(row["doc_id"]),
                int(p_start) if p_start else None,
                int(p_end) if p_end else None,
                query_terms,
                query,
            )
        evidence_p_start = ev.get("paragraph_start") or p_start
        evidence_p_end = ev.get("paragraph_end") or p_end
        page_number = ev.get("page_number")
        location = _format_location(row["rel_path"], row["ext"], evidence_p_start, evidence_p_end, page_number=page_number)
        content = clean_text(row["text"])[:1400]
        evidence_quote = ev.get("quote") or content[:520]
        results.append({
            "title": f"本地知识库：{row['rel_path']}",
            "url": file_url(row["path"]),
            "content": content,
            "snippet": content,
            "_source_type": "local_kb",
            "_direct_source": "local_kb",
            "_matched_query": query,
            "_kb_doc_id": row["doc_id"],
            "_kb_chunk_id": row["chunk_id"],
            "_kb_chunk_index": row["chunk_index"],
            "_kb_score": score,
            "_kb_path": row["rel_path"],
            "_kb_ext": row["ext"],
            "_kb_paragraph_start": evidence_p_start,
            "_kb_paragraph_end": evidence_p_end,
            "_kb_char_start": row["char_start"],
            "_kb_char_end": row["char_end"],
            "_kb_location": location,
            "_kb_quote": evidence_quote,
            "published_at": time.strftime("%Y-%m-%d", time.localtime(float(row["mtime"]))),
        })
        if len(results) >= limit:
            break

    return results


def get_kb_status(base_dir: str | Path, docs_dir: str | Path | None = None, db_path: str | Path | None = None) -> dict:
    paths = ensure_kb_layout(base_dir, docs_dir, db_path)
    doc_files = list(iter_documents(paths.docs_dir))
    file_by_path = {}
    for path in doc_files:
        try:
            st = path.stat()
            file_by_path[str(path.resolve())] = (st.st_mtime, st.st_size)
        except Exception:
            continue

    with connect(paths.db_path) as conn:
        rows = conn.execute("SELECT path, mtime, size FROM documents").fetchall()
        doc_count = len(rows)
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        para_count = conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        latest = conn.execute("SELECT MAX(indexed_at) FROM documents").fetchone()[0]

    indexed_by_path = {row["path"]: row for row in rows}
    new_or_changed = 0
    for path_text, stat_pair in file_by_path.items():
        row = indexed_by_path.get(path_text)
        if not row:
            new_or_changed += 1
            continue
        if float(row["mtime"]) != float(stat_pair[0]) or int(row["size"]) != int(stat_pair[1]):
            new_or_changed += 1

    removed = sum(1 for path_text in indexed_by_path if path_text not in file_by_path)

    return {
        "ok": True,
        "documents_dir": str(paths.docs_dir),
        "db_path": str(paths.db_path),
        "supported_exts": sorted(SUPPORTED_EXTS),
        "file_count": len(doc_files),
        "indexed_documents": int(doc_count or 0),
        "indexed_chunks": int(chunk_count or 0),
        "indexed_paragraphs": int(para_count or 0),
        "last_indexed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest)) if latest else "",
        "new_or_changed_files": int(new_or_changed),
        "removed_files": int(removed),
        "is_stale": bool(new_or_changed or removed),
    }

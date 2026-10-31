from __future__ import annotations

import hashlib
import ipaddress
import logging
import re
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

_ALLOWED_SCHEMES = {"http", "https"}
_FETCH_TIMEOUT = 15
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024


def _is_private_host(hostname: str) -> bool:
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(hostname))
    except (socket.gaierror, ValueError):
        return True
    return any(addr in net for net in _BLOCKED_NETWORKS)


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme '{parsed.scheme}' is not allowed. Use http or https.")
    if not parsed.netloc:
        raise ValueError("URL has no host.")
    if _is_private_host(parsed.hostname or ""):
        raise ValueError(f"URL resolves to a private/internal address and cannot be fetched.")


def _extract_heading_from_markdown(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line.startswith("## "):
            return line[3:].strip()
    return ""


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_doc_id(source_name: str) -> str:
    stem = Path(source_name).stem.lower()
    stem = re.sub(r"[^a-z0-9_]", "_", stem)
    return f"doc_{stem}"


def _build_metadata(source_name: str, doc_id: str, extra: Optional[dict] = None) -> dict:
    base = {
        "source": source_name,
        "doc_id": doc_id,
    }
    if extra:
        base.update(extra)
    return base


def load_markdown(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise ValueError(f"File is empty: {path}")
    doc_id = _make_doc_id(path.name)
    title = _extract_heading_from_markdown(text)
    metadata = _build_metadata(path.name, doc_id, {"title": title})
    logger.info("Loaded Markdown file: %s (%d chars)", path.name, len(text))
    return [Document(page_content=text, metadata=metadata)]


def load_html(path: Path) -> list[Document]:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "head", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError(f"No text content extracted from HTML: {path}")
    doc_id = _make_doc_id(path.name)
    title = soup.find("title")
    title_text = title.get_text(strip=True) if title else ""
    metadata = _build_metadata(path.name, doc_id, {"title": title_text})
    logger.info("Loaded HTML file: %s (%d chars)", path.name, len(text))
    return [Document(page_content=text, metadata=metadata)]


def load_pdf(path: Path) -> list[Document]:
    import pypdf

    docs = []
    reader = pypdf.PdfReader(str(path))
    doc_id = _make_doc_id(path.name)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        metadata = _build_metadata(path.name, doc_id, {"page": i + 1})
        docs.append(Document(page_content=text, metadata=metadata))
    if not docs:
        raise ValueError(f"No text extracted from PDF: {path}")
    logger.info("Loaded PDF: %s (%d pages with text)", path.name, len(docs))
    return docs


def load_url(url: str) -> list[Document]:
    _validate_url(url)

    parsed = urlparse(url)
    source_name = Path(parsed.path).name or parsed.netloc

    headers = {"User-Agent": "RAG-Docs-Assistant/1.0 (+research; not a scraper)"}
    response = requests.get(
        url, headers=headers, timeout=_FETCH_TIMEOUT,
        stream=True,
    )
    response.raise_for_status()

    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise ValueError(f"Response from {url} exceeds {_MAX_RESPONSE_BYTES // 1024 // 1024} MB limit.")
        chunks.append(chunk)
    content_bytes = b"".join(chunks)

    content_type = response.headers.get("Content-Type", "")
    doc_id = _make_doc_id(source_name)

    if "html" in content_type:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content_bytes.decode("utf-8", errors="replace"), "html.parser")
        for tag in soup(["script", "style", "head", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else ""
        metadata = _build_metadata(source_name, doc_id, {"url": url, "title": title_text})
    else:
        text = content_bytes.decode("utf-8", errors="replace").strip()
        title = _extract_heading_from_markdown(text)
        metadata = _build_metadata(source_name, doc_id, {"url": url, "title": title})

    if not text:
        raise ValueError(f"No text content retrieved from URL: {url}")

    logger.info("Loaded URL: %s (%d chars)", url, len(text))
    return [Document(page_content=text, metadata=metadata)]


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext in {".md", ".txt", ".rst"}:
        return load_markdown(path)
    elif ext in {".html", ".htm"}:
        return load_html(path)
    elif ext == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .md, .txt, .rst, .html, .htm, .pdf"
        )


def load_corpus(corpus_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file():
            try:
                loaded = load_file(path)
                docs.extend(loaded)
            except ValueError as e:
                logger.warning("Skipping %s: %s", path.name, e)
    logger.info("Corpus loaded: %d documents from %s", len(docs), corpus_dir)
    return docs


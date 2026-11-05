"""Keep ingestion independent of speech engines and deployment."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n\n")
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())


def extract(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig")
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF support requires: uv sync --extra pdf") from exc
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF is password protected; provide an unlocked copy.")
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in {".mobi", ".epub", ".azw", ".azw3", ".docx", ".rtf", ".html", ".htm"}:
        converter = shutil.which("ebook-convert")
        if converter is None:
            raise ValueError("Install Calibre and ensure ebook-convert is on PATH for this format.")
        with tempfile.TemporaryDirectory(prefix="audiobook-") as directory:
            target = Path(directory) / "book.txt"
            try:
                result = subprocess.run(
                    [converter, str(path.resolve()), str(target)],
                    capture_output=True, text=True, timeout=600,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValueError("Calibre conversion exceeded 10 minutes.") from exc
            if result.returncode:
                raise ValueError(f"Calibre conversion failed (DRM-protected books are unsupported):\n{result.stderr[-2000:]}")
            text = target.read_text(encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    text = normalize(text)
    if not text:
        raise ValueError("No readable text found. Scanned PDFs need OCR before conversion.")
    return text


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    if not 100 <= max_chars <= 4000:
        raise ValueError("Chunk size must be between 100 and 4000 characters.")
    # Prefer paragraphs, then sentence/word boundaries. Even an unbroken token is bounded.
    chunks: list[str] = []
    for paragraph in text.split("\n\n"):
        rest = paragraph.strip()
        while rest:
            end = min(len(rest), max_chars)
            if len(rest) > max_chars:
                sentence_ends = list(re.finditer(r'[.!?][\u201d"\u2019\']?\s+', rest[:max_chars + 1]))
                if sentence_ends:
                    end = sentence_ends[-1].end()
                else:
                    end = rest.rfind(" ", 0, max_chars + 1)
                    if end <= 0:
                        end = max_chars
            chunks.append(rest[:end].strip())
            rest = rest[end:].strip()
    return chunks

"""Convert a scanned PDF dictionary into an EPUB using the Gemini Vision API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import html
import json
import os
import re
import threading
import time
from pathlib import Path

import pymupdf
from ebooklib import epub
from google import genai
from google.genai import errors as genai_errors
from google.genai import types


DEFAULT_MODEL = "gemini-3.5-flash-lite"
CACHE_DIR_NAME = ".ocr_cache"

OCR_PROMPT = """Transcribe this scanned page from a 1935 Turkish-to-Ottoman-Turkish dictionary ("Türkçeden Osmanlıcaya Cep Kılavuzu").

Rules:
1. Preserve every character accurately, including Turkish letters (ç, ğ, ı, İ, ö, ş, ü, â, î, û) and French accents (é, è, ê, à, ç, etc.).
2. Output clean HTML body fragments ONLY (no <html>, <body>, or markdown code fences).
3. For dictionary entries: wrap each entry in <p class="entry">...</p>, make the headword(s) before "=" <strong>...</strong>, and italicize cross-reference targets like <em>aşağılama</em> if italicized on the page.
4. Join words split by a hyphen at the end of a line (e.g., "S'a-\\nbaisser" -> "S'abaisser").
5. Omit the small running guide-word at the top corner (e.g., "Alf") and the page number at the bottom.
6. For title/preface/front-matter pages, use appropriate <h1>, <h2>, or <p> tags.
7. If the page is completely blank, output an empty string.
"""


def load_api_key(env_path: Path) -> str:
    """Load GEMINI_API_KEY from environment or a local .env file."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip().strip("\"'")

    if env_path.exists():
        raw = env_path.read_text(encoding="utf-8")
        match = re.search(r"GEMINI_API_KEY\s*[:=]\s*(\S+)", raw)
        if match:
            return match.group(1).strip().strip("\"'")

    raise RuntimeError(
        "GEMINI_API_KEY not found. Set it in .env or as an environment variable."
    )


def find_pdf(explicit_path: str | None) -> Path:
    """Locate the target PDF in the working directory."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        return path

    pdfs = glob.glob("*.pdf")
    if not pdfs:
        raise FileNotFoundError("No PDF file found in the current directory.")
    return Path(pdfs[0])


def strip_code_fences(text: str) -> str:
    """Remove accidental markdown code fences from model output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:html)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class RateLimiter:
    """Enforces a strict maximum requests-per-minute (RPM) pacing across threads."""

    def __init__(self, max_rpm: float) -> None:
        self._min_interval = 60.0 / max_rpm
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                sleep_for = self._next_allowed - now
                self._next_allowed += self._min_interval
            else:
                sleep_for = 0.0
                self._next_allowed = now + self._min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def ocr_single_page(
    client: genai.Client,
    model: str,
    page_idx: int,
    png_bytes: bytes,
    cache_file: Path,
    rate_limiter: RateLimiter,
    max_retries: int = 5,
) -> tuple[int, str]:
    """OCR a single rendered PDF page, using a JSON cache for resumability."""
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return page_idx, data["html"]

    delay = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            rate_limiter.wait()
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    OCR_PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            html_fragment = strip_code_fences(resp.text or "")
            cache_file.write_text(
                json.dumps({"page": page_idx, "html": html_fragment}, ensure_ascii=False),
                encoding="utf-8",
            )
            return page_idx, html_fragment
        except genai_errors.ClientError as exc:
            if exc.code in (400, 401, 403, 404):
                raise RuntimeError(f"Page {page_idx} client error ({exc.code}): {exc}") from exc
            print(f"Page {page_idx} attempt {attempt} failed ({exc}); retrying in {delay}s...", flush=True)
            if attempt == max_retries:
                raise RuntimeError(f"Page {page_idx} failed after {max_retries} attempts: {exc}") from exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        except Exception as exc:
            print(f"Page {page_idx} attempt {attempt} failed ({exc}); retrying in {delay}s...", flush=True)
            if attempt == max_retries:
                raise RuntimeError(f"Page {page_idx} failed after {max_retries} attempts: {exc}") from exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    return page_idx, ""


def build_epub(
    pdf_path: Path,
    page_htmls: dict[int, str],
    cover_png: bytes | None,
    output_path: Path,
    pages_per_chapter: int = 20,
) -> None:
    """Assemble transcribed HTML pages into a styled EPUB."""
    book = epub.EpubBook()
    book.set_identifier("tdk-turkceden-osmanlicaya-cep-kilavuzu-1935")
    book.set_title("Türkçeden Osmanlıcaya Cep Kılavuzu")
    book.set_language("tr")
    book.add_author("Türk Dil Kurumu (T.D.K.)")

    if cover_png:
        book.set_cover("cover.png", cover_png)

    css = epub.EpubItem(
        uid="style_default",
        file_name="style/default.css",
        media_type="text/css",
        content="""
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.45;
    margin: 1em;
}
h1, h2 {
    text-align: center;
    margin-top: 1em;
    margin-bottom: 0.6em;
}
p.entry {
    margin: 0.25em 0;
    padding-left: 1.2em;
    text-indent: -1.2em;
}
hr.page-break {
    border: none;
    border-top: 1px solid #ddd;
    margin: 0.8em 0;
}
""".strip().encode("utf-8"),
    )
    book.add_item(css)

    sorted_pages = sorted(page_htmls.keys())
    chapters: list[epub.EpubHtml] = []

    for chunk_start in range(0, len(sorted_pages), pages_per_chapter):
        chunk = sorted_pages[chunk_start : chunk_start + pages_per_chapter]
        first_p, last_p = chunk[0] + 1, chunk[-1] + 1
        title = f"Sayfa {first_p}–{last_p}" if first_p != last_p else f"Sayfa {first_p}"
        file_name = f"chap_{chunk_start // pages_per_chapter + 1:03d}.xhtml"

        body_parts = []
        for idx in chunk:
            content = page_htmls[idx].strip()
            if content:
                body_parts.append(f'<div class="pdf-page" data-page="{idx + 1}">\n{content}\n</div>')

        if not body_parts:
            continue

        chap = epub.EpubHtml(title=title, file_name=file_name, lang="tr")
        chap.content = (
            f"<html><head><title>{html.escape(title)}</title></head>"
            f"<body>\n" + '\n<hr class="page-break"/>\n'.join(body_parts) + "\n</body></html>"
        ).encode("utf-8")
        chap.add_item(css)
        book.add_item(chap)
        chapters.append(chap)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters]

    epub.write_epub(str(output_path), book)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert scanned PDF to EPUB via Gemini Vision.")
    parser.add_argument("--pdf", help="Path to input PDF (defaults to first PDF in cwd).")
    parser.add_argument("--output", help="Output EPUB path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI for PDF pages (default: 200).")
    parser.add_argument("--workers", type=int, default=2, help="Parallel API workers (default: 2).")
    parser.add_argument("--rpm", type=float, default=12.0, help="Max requests per minute (default: 12).")
    parser.add_argument("--pages", help="Comma-separated 0-based page indices or ranges (e.g. 0,2,20-25).")
    return parser.parse_args()


def parse_page_selection(spec: str | None, total_pages: int) -> list[int]:
    if not spec:
        return list(range(total_pages))
    selected: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            selected.extend(range(int(start_s), int(end_s) + 1))
        else:
            selected.append(int(part))
    return [p for p in selected if 0 <= p < total_pages]


def main() -> None:
    args = parse_args()
    api_key = load_api_key(Path(".env"))
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60_000),
    )

    pdf_path = find_pdf(args.pdf)
    output_path = Path(args.output) if args.output else pdf_path.with_suffix(".epub")
    cache_dir = Path(CACHE_DIR_NAME)
    cache_dir.mkdir(exist_ok=True)

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    target_pages = parse_page_selection(args.pages, total_pages)
    print(f"PDF: {pdf_path.name} ({total_pages} pages). Processing {len(target_pages)} page(s) with {args.model}...", flush=True)

    cover_png = doc[0].get_pixmap(dpi=args.dpi).tobytes("png") if total_pages > 0 else None

    rendered: dict[int, bytes] = {}
    for idx in target_pages:
        cache_file = cache_dir / f"page_{idx:04d}.json"
        if not cache_file.exists():
            rendered[idx] = doc[idx].get_pixmap(dpi=args.dpi).tobytes("png")

    page_htmls: dict[int, str] = {}
    rate_limiter = RateLimiter(args.rpm)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                ocr_single_page,
                client,
                args.model,
                idx,
                rendered.get(idx, b""),
                cache_dir / f"page_{idx:04d}.json",
                rate_limiter,
            ): idx
            for idx in target_pages
        }
        for future in as_completed(futures):
            idx, text = future.result()
            page_htmls[idx] = text
            completed += 1
            if completed % 10 == 0 or completed == len(target_pages):
                print(f"Progress: {completed}/{len(target_pages)} pages")

    build_epub(pdf_path, page_htmls, cover_png, output_path)
    print(f"Saved EPUB: {output_path}")


if __name__ == "__main__":
    main()

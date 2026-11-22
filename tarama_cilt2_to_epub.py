"""Convert Tarama Dergisi Cilt 2 (2-column scanned PDF) to a polished EPUB and dataset."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
import statistics
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf
from ebooklib import epub

from vision_to_epub import (
    VISION_ENDPOINT,
    append_dehyphenated,
    get_access_token,
    make_chapter,
    resolve_project_id,
)


DEFAULT_PDF = "Osmanlıcadan Türkçeye Söz Karşılıkları Tarama Dergisi Cilt 2.pdf"
CACHE_DIR_NAME = ".vision_cache_td2"

TURKISH_ALPHABET = [
    "A", "B", "C", "Ç", "D", "E", "F", "G", "H", "I", "İ", "J",
    "K", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U",
    "Ü", "V", "Y", "Z",
]
ASCII_SLUG = {"Ç": "ch", "İ": "ii", "Ö": "oe", "Ş": "sh", "Ü": "ue"}


def extract_column_lines(annotation: dict, single_column: bool = False) -> list[str]:
    """Reconstruct clean reading-order lines from Cloud Vision word bounding boxes."""
    if single_column or not annotation.get("pages"):
        return [ln.strip() for ln in annotation.get("text", "").splitlines() if ln.strip()]

    page = annotation["pages"][0]
    width = float(page.get("width") or 1)
    height = float(page.get("height") or 1)

    words: list[tuple[float, float, float, float, str]] = []
    for block in page.get("blocks", []):
        for para in block.get("paragraphs", []):
            for word in para.get("words", []):
                text = "".join(s.get("text", "") for s in word.get("symbols", [])).strip()
                if not text:
                    continue
                verts = word.get("boundingBox", {}).get("vertices", [])
                if not verts:
                    continue
                xs = [float(v.get("x", 0)) for v in verts]
                ys = [float(v.get("y", 0)) for v in verts]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                cx = (x0 + x1) / 2.0
                # Skip vertical center divider rule if OCR'd as '|'
                if text == "|" and 0.44 * width <= cx <= 0.56 * width:
                    continue
                words.append((x0, x1, y0, y1, text))

    if not words:
        return []

    heights = [y1 - y0 for _, _, y0, y1, _ in words if y1 > y0]
    med_h = statistics.median(heights) if heights else 14.0

    # Find the cleanest vertical gutter near the page center (46%..54% of width)
    best_x = width * 0.50
    best_overlap = float("inf")
    for pct in range(46, 55):
        cand_x = width * (pct / 100.0)
        # Count words whose horizontal span crosses cand_x (ignoring top header band)
        overlap = sum(
            1
            for x0, x1, y0, y1, _ in words
            if (y0 + y1) / 2.0 > 0.08 * height and x0 < cand_x < x1
        )
        if overlap < best_overlap:
            best_overlap = overlap
            best_x = cand_x

    # Drop top running header band (e.g. "Ant — 870 — Ara")
    body_words = [
        w for w in words
        if (w[2] + w[3]) / 2.0 >= 0.082 * height
    ]
    left_words = [w for w in body_words if (w[0] + w[1]) / 2.0 < best_x]
    right_words = [w for w in body_words if (w[0] + w[1]) / 2.0 >= best_x]

    def group_column(col_words: list[tuple[float, float, float, float, str]]) -> list[str]:
        if not col_words:
            return []
        col_words.sort(key=lambda w: ((w[2] + w[3]) / 2.0, w[0]))
        lines_words: list[list[tuple[float, float, float, float, str]]] = []
        line_cys: list[float] = []
        y_tol = max(med_h * 0.55, 6.0)
        for w in col_words:
            cy = (w[2] + w[3]) / 2.0
            if not lines_words or abs(cy - line_cys[-1]) > y_tol:
                lines_words.append([w])
                line_cys.append(cy)
            else:
                lines_words[-1].append(w)
                line_cys[-1] = sum((x[2] + x[3]) / 2.0 for x in lines_words[-1]) / len(lines_words[-1])

        out: list[str] = []
        for lw in lines_words:
            lw.sort(key=lambda w: w[0])
            line_str = " ".join(w[4] for w in lw)
            # Clean punctuation spacing produced by word joining
            line_str = re.sub(r"\s+([,.;:!?\]])", r"\1", line_str)
            line_str = re.sub(r"([\[])\s+", r"\1", line_str)
            out.append(line_str.strip())
        return out

    return group_column(left_words) + group_column(right_words)


def ocr_batch_structured(
    token: str,
    project_id: str,
    batch_items: list[tuple[int, bytes, bool]],
) -> dict[int, list[str]]:
    """Call Cloud Vision API and return column-sorted lines per page."""
    requests_payload = [
        {
            "image": {"content": base64.b64encode(png_bytes).decode("ascii")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {"languageHints": ["tr", "fr"]},
        }
        for _, png_bytes, _ in batch_items
    ]
    body = json.dumps({"requests": requests_payload}).encode("utf-8")
    req = urllib.request.Request(
        VISION_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": project_id,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vision API HTTP {exc.code}: {err_body[:500]}") from exc

    results: dict[int, list[str]] = {}
    for (page_idx, _, single_col), item in zip(batch_items, data.get("responses", [])):
        if "error" in item:
            raise RuntimeError(f"Page {page_idx} Vision API error: {item['error']}")
        ann = item.get("fullTextAnnotation", {})
        results[page_idx] = extract_column_lines(ann, single_column=single_col)
    return results


def normalize_turkish_initial(headword: str, current_letter: str) -> str:
    """Map an entry's first character to its canonical Turkish alphabet chapter."""
    cleaned = headword.lstrip("-«\"' ").strip()
    if not cleaned:
        return current_letter
    ch = cleaned[0]
    upper_map = {
        "a": "A", "â": "A", "Â": "A",
        "b": "B", "c": "C", "ç": "Ç", "d": "D", "e": "E", "f": "F",
        "g": "G", "ğ": "G", "Ğ": "G", "h": "H",
        "ı": "I", "i": "İ", "î": "İ", "Î": "İ",
        "j": "J", "k": "K", "l": "L", "m": "M", "n": "N",
        "o": "O", "ö": "Ö", "p": "P", "r": "R", "s": "S", "ş": "Ş",
        "t": "T", "u": "U", "û": "U", "Û": "U", "ü": "Ü", "v": "V",
        "y": "Y", "z": "Z",
    }
    cand = upper_map.get(ch, ch.upper())
    if cand in TURKISH_ALPHABET:
        return cand
    return current_letter


def clean_prose_page(lines: list[str], title_to_strip: str) -> list[str]:
    """Reflow a single-column front/back matter page into clean paragraphs."""
    filtered: list[str] = []
    for ln in lines:
        if ln.upper().replace("İ", "I") == title_to_strip.upper().replace("İ", "I"):
            continue
        if re.fullmatch(r"[\-—–\s]*\d{1,4}[\-—–\s]*", ln):
            continue
        if re.fullmatch(r"[IVXivx]{1,5}", ln):
            continue
        filtered.append(ln)

    joined = ""
    for ln in filtered:
        joined = append_dehyphenated(joined, ln)
    joined = re.sub(r"\s+", " ", joined).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ«])", joined)
    paras: list[str] = []
    chunk: list[str] = []
    for s in sentences:
        chunk.append(s.strip())
        if len(chunk) >= 4:
            paras.append(" ".join(chunk))
            chunk = []
    if chunk:
        paras.append(" ".join(chunk))
    return paras


def extract_dictionary_entries(page_lines: dict[int, list[str]]) -> list[dict[str, str]]:
    """Merge 2-column pages 3..457 into structured entries grouped by Turkish letter."""
    raw_entries: list[str] = []

    for page_idx in range(3, 458):
        lines = page_lines.get(page_idx, [])
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            # Skip standalone page numbers / header artifacts / single-letter section headings
            if re.fullmatch(r"[\-—–\s]*\d{1,4}[\-—–\s]*", s):
                continue
            if re.fullmatch(r"[A-ZÇĞİÖŞÜ]", s):
                continue

            prev_text = raw_entries[-1] if raw_entries else ""
            prev_complete = bool(prev_text and prev_text.rstrip().endswith((".", "!", "?")))

            starts_new = False
            if not raw_entries:
                starts_new = True
            elif prev_complete and s[0].isupper():
                starts_new = True
            elif (
                not prev_text.endswith(("-", ","))
                and re.match(r"^[A-ZÇĞİÖŞÜ][a-zçğıöşüâîû\s\[\]\?!\-]{0,35},\s+[A-ZÇĞİÖŞÜ]", s)
            ):
                starts_new = True

            if starts_new:
                raw_entries.append(s)
            else:
                raw_entries[-1] = append_dehyphenated(raw_entries[-1], s)

    # First parse headwords/equivalents, then assign monotonic A-Z chapters using a 3-entry lookahead
    parsed: list[tuple[str, str, str]] = []
    for entry in raw_entries:
        text = re.sub(r"\s+", " ", entry).strip()
        m = re.match(r"^([^,]+),\s*(.*)$", text)
        if m:
            headword = m.group(1).strip().rstrip("·•")
            equivalents = m.group(2).strip()
        else:
            parts = text.split(None, 1)
            headword = parts[0] if parts else ""
            equivalents = parts[1] if len(parts) > 1 else ""
        parsed.append((headword, equivalents, text))

    current_letter = "A"
    records: list[dict[str, str]] = []
    n = len(parsed)
    for i, (headword, equivalents, text) in enumerate(parsed):
        cand = normalize_turkish_initial(headword, current_letter)
        cur_i = TURKISH_ALPHABET.index(current_letter)
        cand_i = TURKISH_ALPHABET.index(cand) if cand in TURKISH_ALPHABET else cur_i

        # Advance chapter monotonically only when confirmed by consecutive entries
        if cand_i > cur_i and cand_i <= cur_i + 2:
            needed = 1 if cand == "J" else 3
            window = parsed[i : min(n, i + needed)]
            if len(window) == needed and all(
                normalize_turkish_initial(w[0], cand) == cand for w in window
            ):
                current_letter = cand

        records.append(
            {
                "letter": current_letter,
                "headword": headword,
                "equivalents": equivalents,
                "raw_entry": text,
            }
        )
    return records


def build_epub_and_export(
    page_lines: dict[int, list[str]],
    cover_png: bytes | None,
    output_epub: Path,
    export_dir: Path | None,
) -> int:
    """Build polished A–Z EPUB and optional JSON/CSV dataset."""
    book = epub.EpubBook()
    book.set_identifier("tdtc-tarama-dergisi-cilt-2-1934")
    book.set_title("Osmanlıcadan Türkçeye Söz Karşılıkları: Tarama Dergisi (Cilt 2)")
    book.set_language("tr")
    book.add_author("Türk Dili Tetkik Cemiyeti (T.D.T.C.)")

    if cover_png:
        book.set_cover("cover.png", cover_png)

    css = epub.EpubItem(
        uid="style_default",
        file_name="style/default.css",
        media_type="text/css",
        content="""
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.5;
    margin: 1.2em;
}
h1, h2 {
    text-align: center;
    margin-top: 0.8em;
    margin-bottom: 0.8em;
}
h1.letter-heading {
    font-size: 2em;
    border-bottom: 1px solid #888;
    padding-bottom: 0.25em;
    margin-bottom: 0.8em;
}
.title-page {
    text-align: center;
    margin-top: 2em;
}
.title-page .org {
    letter-spacing: 0.15em;
    font-weight: bold;
    margin-bottom: 1.5em;
}
.title-page .subtitle {
    font-size: 1.1em;
    margin-top: 0.8em;
}
.title-page .imprint {
    margin-top: 2.5em;
    line-height: 1.6;
}
p.prose {
    text-indent: 1.2em;
    margin: 0.45em 0;
    text-align: justify;
}
p.entry {
    margin: 0.28em 0;
    padding-left: 1.4em;
    text-indent: -1.4em;
}
""".strip().encode("utf-8"),
    )
    book.add_item(css)

    chapters: list[epub.EpubHtml] = []

    title_html = """<div class="title-page">
  <p class="org">T. D. T. C.</p>
  <p class="subtitle">Osmanlıcadan Türkçeye Söz Karşılıkları</p>
  <h1>TARAMA DERGİSİ</h1>
  <p class="subtitle"><strong>Cilt: 2</strong><br/>Türkçeden Osmanlıcaya İndeks</p>
  <p class="imprint">İstanbul — Devlet Matbaası<br/>1934</p>
</div>"""
    chapters.append(make_chapter("Kapak / Künye", "00_kunye.xhtml", title_html, css))

    if 1 in page_lines:
        paras = clean_prose_page(page_lines[1], "ÖNSÖZ")
        body = "<h1>Önsöz</h1>\n" + "\n".join(f'<p class="prose">{html.escape(p)}</p>' for p in paras)
        chapters.append(make_chapter("Önsöz", "01_onsoz.xhtml", body, css))

    if 2 in page_lines:
        paras = clean_prose_page(page_lines[2], "İNDEKS NASIL HAZIRLANDI?")
        body = "<h1>İndeks Nasıl Hazırlandı?</h1>\n" + "\n".join(
            f'<p class="prose">{html.escape(p)}</p>' for p in paras
        )
        chapters.append(make_chapter("İndeks Nasıl Hazırlandı?", "02_indeks_nasil_hazirlandi.xhtml", body, css))

    records = extract_dictionary_entries(page_lines)
    by_letter: dict[str, list[dict[str, str]]] = {ltr: [] for ltr in TURKISH_ALPHABET}
    for rec in records:
        by_letter.setdefault(rec["letter"], []).append(rec)

    for idx, ltr in enumerate(TURKISH_ALPHABET, start=1):
        items = by_letter.get(ltr, [])
        if not items:
            continue
        entry_htmls = []
        for rec in items:
            head = html.escape(rec["headword"])
            eq = html.escape(rec["equivalents"])
            if eq:
                entry_htmls.append(f'<p class="entry"><strong>{head}</strong>, {eq}</p>')
            else:
                entry_htmls.append(f'<p class="entry"><strong>{head}</strong></p>')
        body = f'<h1 class="letter-heading">{html.escape(ltr)}</h1>\n' + "\n".join(entry_htmls)
        slug = ASCII_SLUG.get(ltr, ltr.lower())
        chapters.append(make_chapter(ltr, f"dict_{idx:02d}_{slug}.xhtml", body, css))

    if 458 in page_lines:
        paras = clean_prose_page(page_lines[458], "BİTİRİRKEN")
        body = "<h1>Bitirirken</h1>\n" + "\n".join(f'<p class="prose">{html.escape(p)}</p>' for p in paras)
        chapters.append(make_chapter("Bitirirken", "99_bitirirken.xhtml", body, css))

    for chap in chapters:
        book.add_item(chap)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters]
    epub.write_epub(str(output_epub), book)

    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        stem = "tarama_dergisi_cilt2_1934"
        (export_dir / f"{stem}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (export_dir / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["letter", "headword", "equivalents", "raw_entry"])
            writer.writeheader()
            writer.writerows(records)

    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Tarama Dergisi Cilt 2 to EPUB via Cloud Vision.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Input PDF path.")
    parser.add_argument("--output", help="Output EPUB path.")
    parser.add_argument("--project", help="GCP project ID.")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150).")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per Vision API request (max 16).")
    parser.add_argument("--export-dir", help="Optional directory to export JSON/CSV dataset.")
    parser.add_argument("--pages", help="Optional 0-based page selection (e.g. 20).")
    parser.add_argument("--no-cache", action="store_true", help="Ignore existing cache files for selected pages.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_epub = Path(args.output) if args.output else pdf_path.with_suffix(".epub")
    cache_dir = Path(CACHE_DIR_NAME)
    cache_dir.mkdir(exist_ok=True)

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    if args.pages:
        target_pages = [int(p.strip()) for p in args.pages.split(",")]
    else:
        target_pages = list(range(total_pages))

    print(f"PDF: {pdf_path.name} ({total_pages} pages). Processing {len(target_pages)} page(s)...", flush=True)
    cover_png = doc[0].get_pixmap(dpi=args.dpi).tobytes("png")

    page_lines: dict[int, list[str]] = {}
    uncached: list[int] = []
    for idx in target_pages:
        cf = cache_dir / f"page_{idx:04d}.json"
        if cf.exists() and not args.no_cache:
            data = json.loads(cf.read_text(encoding="utf-8"))
            if "lines" in data:
                page_lines[idx] = data["lines"]
                continue
        uncached.append(idx)

    if uncached:
        project_id = resolve_project_id(args.project)
        token = get_access_token()
        for start in range(0, len(uncached), args.batch_size):
            batch_indices = uncached[start : start + args.batch_size]
            batch_items = [
                (idx, doc[idx].get_pixmap(dpi=args.dpi).tobytes("png"), idx in (0, 1, 2, 458))
                for idx in batch_indices
            ]
            results = ocr_batch_structured(token, project_id, batch_items)
            for idx, lines in results.items():
                page_lines[idx] = lines
                (cache_dir / f"page_{idx:04d}.json").write_text(
                    json.dumps({"page": idx, "lines": lines}, ensure_ascii=False),
                    encoding="utf-8",
                )
            print(f"Progress: {len(page_lines)}/{len(target_pages)} pages", flush=True)
    else:
        print(f"All pages loaded from {CACHE_DIR_NAME} (0 API calls).", flush=True)

    if not args.pages:
        count = build_epub_and_export(
            page_lines,
            cover_png,
            output_epub,
            Path(args.export_dir) if args.export_dir else None,
        )
        print(f"Saved EPUB: {output_epub} ({count} dictionary entries)", flush=True)
    else:
        for idx in target_pages:
            print(f"\n=== Page {idx} ({len(page_lines[idx])} lines) ===")
            for ln in page_lines[idx][:35]:
                print(" ", ln)


if __name__ == "__main__":
    main()

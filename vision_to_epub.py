"""Convert a scanned PDF dictionary into a polished EPUB using Google Cloud Vision API."""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import html
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pymupdf
from ebooklib import epub


VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
CACHE_DIR_NAME = ".vision_cache"

DICTIONARY_LETTERS: list[tuple[str, int, int]] = [
    ("A", 10, 43),
    ("B", 44, 77),
    ("C", 78, 81),
    ("Ç", 82, 93),
    ("D", 94, 118),
    ("E", 119, 133),
    ("F", 134, 135),
    ("G", 136, 156),
    ("H", 157, 165),
    ("I", 166, 167),
    ("İ", 168, 183),
    ("J", 184, 184),
    ("K", 185, 217),
    ("L", 218, 219),
    ("M", 220, 223),
    ("N", 224, 226),
    ("O", 227, 234),
    ("Ö", 235, 245),
    ("P", 246, 249),
    ("R", 250, 251),
    ("S", 252, 279),
    ("Ş", 280, 283),
    ("T", 284, 300),
    ("U", 301, 310),
    ("Ü", 311, 315),
    ("V", 316, 319),
    ("Y", 320, 348),
    ("Z", 349, 351),
]

# Allowed initial characters (including common OCR confusions) per dictionary letter.
LETTER_INITIALS: dict[str, set[str]] = {
    "A": {"A", "a"},
    "B": {"B", "b"},
    "C": {"C", "c", "Ç"},
    "Ç": {"Ç", "C", "ç"},
    "D": {"D", "d"},
    "E": {"E", "e"},
    "F": {"F", "f"},
    "G": {"G", "Ğ", "g"},
    "H": {"H", "h"},
    "I": {"I", "İ", "l", "1"},
    "İ": {"İ", "I", "i", "l"},
    "J": {"J", "j"},
    "K": {"K", "k"},
    "L": {"L", "l"},
    "M": {"M", "m"},
    "N": {"N", "n"},
    "O": {"O", "Ö", "o"},
    "Ö": {"Ö", "O", "ö"},
    "P": {"P", "p"},
    "R": {"R", "r"},
    "S": {"S", "Ş", "s"},
    "Ş": {"Ş", "S", "$", "ş"},
    "T": {"T", "t"},
    "U": {"U", "Ü", "u"},
    "Ü": {"Ü", "U", "ü"},
    "V": {"V", "v"},
    "Y": {"Y", "y"},
    "Z": {"Z", "z"},
}

ROMAN_NUMERAL_ARTIFACTS = {
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "x", "TV", "VIN", "☑",
}


def find_gcloud() -> str:
    """Locate gcloud executable on Windows or PATH."""
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", ""))
        / "Google"
        / "Cloud SDK"
        / "google-cloud-sdk"
        / "bin"
        / "gcloud.cmd",
        Path(r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
        Path(r"C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return "gcloud"


def get_access_token() -> str:
    """Fetch a fresh OAuth2 access token from gcloud."""
    gcloud = find_gcloud()
    result = subprocess.run(
        [gcloud, "auth", "print-access-token"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def resolve_project_id(explicit: str | None) -> str:
    """Resolve GCP project ID from CLI flag, env var, or active gcloud config."""
    if explicit:
        return explicit.strip()
    env_proj = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if env_proj:
        return env_proj
    gcloud = find_gcloud()
    result = subprocess.run(
        [gcloud, "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=False,
    )
    proj = result.stdout.strip()
    if proj and proj != "(unset)":
        return proj
    raise RuntimeError(
        "No GCP project ID configured. Pass --project, set GOOGLE_CLOUD_PROJECT, or run 'gcloud config set project <ID>'."
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


def ocr_batch(
    token: str,
    project_id: str,
    batch_items: list[tuple[int, bytes]],
) -> dict[int, str]:
    """Send up to 16 page images in a single Cloud Vision API batch request."""
    requests_payload = [
        {
            "image": {"content": base64.b64encode(png_bytes).decode("ascii")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {"languageHints": ["tr", "fr"]},
        }
        for _, png_bytes in batch_items
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
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vision API HTTP {exc.code}: {err_body[:500]}") from exc

    responses = data.get("responses", [])
    results: dict[int, str] = {}
    for (page_idx, _), item in zip(batch_items, responses):
        if "error" in item:
            raise RuntimeError(f"Page {page_idx} Vision API error: {item['error']}")
        text = item.get("fullTextAnnotation", {}).get("text", "")
        results[page_idx] = text
    return results


def append_dehyphenated(buf: str, nxt: str) -> str:
    """Join two lines, removing a trailing word-break hyphen when appropriate."""
    if not buf:
        return nxt
    if buf.endswith("-") and not buf.endswith(" -") and not buf.endswith("=-"):
        # Only join without space when the continuation begins with a lowercase letter
        # (avoids fusing a suffix/prefix marker like 'Hem-' onto a capitalized word).
        if nxt and nxt[0].islower():
            return buf[:-1] + nxt
    return buf + " " + nxt


def strip_front_page_lines(raw_text: str) -> list[str]:
    """Clean standalone Roman numerals and OCR noise from front-matter pages."""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    cleaned: list[str] = []
    for ln in lines:
        if ln in ROMAN_NUMERAL_ARTIFACTS:
            continue
        if re.fullmatch(r"[\.\-,;:|☑]+", ln):
            continue
        if re.fullmatch(r"\d{1,3}", ln):
            continue
        cleaned.append(ln)
    return cleaned


def reflow_prose_lines(lines: list[str], max_sentences_per_para: int = 4) -> list[str]:
    """Join wrapped prose lines across pages into clean, de-hyphenated paragraphs."""
    full_text = ""
    for ln in lines:
        full_text = append_dehyphenated(full_text, ln)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    # Split into sentences and group cleanly into readable paragraphs.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])", full_text)
    paragraphs: list[str] = []
    chunk: list[str] = []
    for sent in sentences:
        chunk.append(sent.strip())
        if len(chunk) >= max_sentences_per_para:
            paragraphs.append(" ".join(chunk))
            chunk = []
    if chunk:
        paragraphs.append(" ".join(chunk))
    return paragraphs


def build_title_html() -> str:
    """Return a clean, artifact-free title and imprint page."""
    return """<div class="title-page">
  <p class="org">T. D. K.</p>
  <h1>TÜRKÇEDEN OSMANLICAYA<br/>CEP KILAVUZU</h1>
  <p class="imprint">1935<br/>İstanbul — Devlet Basımevi</p>
  <p class="copyright">Her türlü telif hakkı Türk Dil Kurumunundur.</p>
</div>"""


def build_onsoz_html(page_texts: dict[int, str]) -> str:
    """Format page 2 (Önsöz) into clean prose paragraphs and signature."""
    lines = strip_front_page_lines(page_texts.get(2, ""))
    if lines and lines[0].upper() in {"ÖNSÖZ", "ONSOZ"}:
        lines.pop(0)

    signature_lines: list[str] = []
    if lines and "Türk Dil Kurumu" in lines[-1]:
        signature_lines.append(lines.pop())

    # Split into the two original paragraphs around "Bir endeks olarak"
    split_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Bir endeks")),
        None,
    )
    if split_idx is not None:
        para_groups = [lines[:split_idx], lines[split_idx:]]
    else:
        para_groups = [lines]

    parts = ["<h1>Önsöz</h1>"]
    for grp in para_groups:
        joined = ""
        for ln in grp:
            joined = append_dehyphenated(joined, ln)
        joined = re.sub(r"\s+", " ", joined).strip()
        if joined:
            parts.append(f'<p class="prose">{html.escape(joined)}</p>')

    for sig in signature_lines:
        parts.append(f'<p class="signature">{html.escape(sig)}</p>')
    return "\n".join(parts)


def build_bildirig_html(page_texts: dict[int, str]) -> str:
    """Merge pages 3–7 seamlessly into the T.D.K. declaration chapter."""
    all_lines: list[str] = []
    for idx in range(3, 8):
        lines = strip_front_page_lines(page_texts.get(idx, ""))
        if idx == 3 and lines and "BİLDİRİĞİ" in lines[0].upper():
            lines.pop(0)
        all_lines.extend(lines)

    # Extract trailing signature ("Türk Dil Kurumu / Genel Sekreteri / İbrahim Necmi Dilmen")
    sig_start = None
    for i in range(max(0, len(all_lines) - 5), len(all_lines)):
        if all_lines[i].startswith("Türk Dil Kurumu"):
            sig_start = i
            break

    signature_lines: list[str] = []
    if sig_start is not None:
        signature_lines = all_lines[sig_start:]
        all_lines = all_lines[:sig_start]

    paragraphs = reflow_prose_lines(all_lines, max_sentences_per_para=4)
    parts = ["<h1>Türk Dil Kurumunun Bildiriği</h1>"]
    for para in paragraphs:
        parts.append(f'<p class="prose">{html.escape(para)}</p>')

    if signature_lines:
        sig_html = "<br/>".join(html.escape(s) for s in signature_lines)
        parts.append(f'<p class="signature">{sig_html}</p>')
    return "\n".join(parts)


def build_imler_html(page_texts: dict[int, str]) -> str:
    """Merge pages 8–9 seamlessly into the Symbols and Abbreviations chapter."""
    all_lines: list[str] = []
    for idx in (8, 9):
        lines = strip_front_page_lines(page_texts.get(idx, ""))
        if idx == 8:
            # Drop the 3-line printed heading
            while lines and not re.match(r"^\d+\.", lines[0]):
                lines.pop(0)
        all_lines.extend(lines)

    items: list[tuple[str, str]] = []
    for ln in all_lines:
        if re.match(r"^\d{1,2}\.\s", ln):
            items.append(("rule", ln))
        elif re.match(r"^[a-z]\)\s", ln):
            items.append(("subrule", ln))
        elif items:
            cls, prev = items[-1]
            items[-1] = (cls, append_dehyphenated(prev, ln))
        else:
            items.append(("rule", ln))

    parts = ["<h1>İmler ve Kısaltmalar</h1>"]
    for cls, text in items:
        clean_text = re.sub(r"\s+", " ", text).strip()
        parts.append(f'<p class="{cls}">{html.escape(clean_text)}</p>')
    return "\n".join(parts)


def build_colophon_html(page_texts: dict[int, str]) -> str:
    """Clean page 352 back-matter sales colophon."""
    raw_lines = [ln.strip() for ln in page_texts.get(352, "").splitlines() if ln.strip()]
    kept: list[str] = []
    for ln in raw_lines:
        # Stop after İstanbul / drop short trailing OCR speck (e.g. "uş")
        if len(ln) <= 3 and kept:
            continue
        kept.append(ln)
    parts = ["<h1>Satış Merkezi</h1>", '<div class="colophon">']
    for ln in kept:
        parts.append(f"<p>{html.escape(ln)}</p>")
    parts.append("</div>")
    return "\n".join(parts)


def clean_dictionary_page_lines(raw_text: str) -> list[str]:
    """Strip page numbers, running headers, and OCR noise from a single dictionary page."""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return []

    # Strip bottom page number
    if lines and re.fullmatch(r"\d{1,4}[\.\-]?", lines[-1]):
        lines.pop()

    # Strip top running 3-letter guide-word (2-4 chars, no '=' or '(' or digits)
    if lines and len(lines[0]) <= 4 and "=" not in lines[0] and "(" not in lines[0]:
        if re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşüâîû&ŏę\.\$\-]{1,4}", lines[0]):
            lines.pop(0)

    cleaned: list[str] = []
    for i, ln in enumerate(lines):
        # Collapse runs of '=' into a single '='
        ln = re.sub(r"={2,}", "=", ln)

        # Strip section letter headings / stray 1-character noise (keep standalone '=' and '-')
        if ln in {".", ",", ";", ":", "|", "☑"}:
            continue
        if i < 4 and re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü\$]\.?", ln):
            continue
        if re.fullmatch(r"[A-ZÇĞİÖŞÜ\$]", ln):
            continue

        # Normalize standalone '-' between headword and definition into '=' if prev lacks '='
        if ln == "-":
            if cleaned and "=" not in cleaned[-1] and not re.search(r"\d\.$", cleaned[-1]):
                cleaned.append("=")
            continue

        cleaned.append(ln)

    return cleaned


def starts_with_chapter_letter(ln: str, letter: str) -> bool:
    """Return True if line begins with a headword character matching the chapter."""
    allowed = LETTER_INITIALS.get(letter, {letter})
    first_ch = ln[0]
    if first_ch == "-" and len(ln) > 1 and ln[1].isalpha():
        return True
    return first_ch in allowed


def is_continuation_line(ln: str, prev_entry: str, letter: str) -> bool:
    """Determine whether a line continues the previous dictionary entry."""
    if not prev_entry:
        return False

    # Unfinished previous line indicators
    if prev_entry.endswith("-") and not prev_entry.endswith(" -"):
        # A trailing '-' after a prefix like 'Hem-' followed by a new capitalized entry
        # of this chapter (with '=' or '(Bak:') starts a new entry.
        if (
            "=" in prev_entry
            and starts_with_chapter_letter(ln, letter)
            and ("=" in ln or re.search(r"\(Bak\s*:", ln))
        ):
            return False
        return True
    if prev_entry.endswith(("=", ",", ";", ":", "(")):
        return True
    if re.search(r"\bBak\s*:\s*[^)]*$", prev_entry):
        return True
    if re.search(r"\b\d+\.$", prev_entry):
        return True

    # Explicit continuation line starters
    if ln.startswith(("=", "(", ")", ",", ";", ":")):
        return True
    if re.match(r"^\d+\.", ln):
        return True
    if ln[0].islower():
        return True

    # If it doesn't start with this chapter's letter (e.g. Ottoman/French words), it's a continuation
    if not starts_with_chapter_letter(ln, letter):
        return True

    # Even if it starts with the chapter's letter, if the previous line already had '='
    # and the current line looks like a pure entry headword (has '=' or '(Bak:') -> new entry
    if "=" in ln or re.search(r"\(Bak\s*:", ln):
        return False

    # If prev_entry doesn't yet have '=' and isn't a Bak-only cross-reference, keep attaching
    if "=" not in prev_entry and "(Bak" not in prev_entry:
        return True

    return False


def normalize_entry_text(entry: str) -> str:
    """Collapse repeated '=' artifacts and normalize spacing on a merged entry."""
    text = re.sub(r"={2,}", "=", entry)
    text = re.sub(r"(?:=\s*){2,}", "= ", text)
    text = re.sub(r"\s*=\s*", " = ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lstrip("= ").rstrip("=").strip()


def parse_entry_fields(letter: str, entry: str) -> dict[str, str]:
    """Split a normalized dictionary entry into structured headword, cross-ref, and equivalents."""
    text = normalize_entry_text(entry)
    cross_refs = [m.group(1).strip() for m in re.finditer(r"\(Bak\s*:?\s*([^)]+)\)", text)]

    m = re.match(r"^(.+?)(\s*(?:=|\(Bak\s*:|:\s+|\s+\d+\.\s+).*)$", text)
    if m:
        headword = m.group(1).strip()
        rest = m.group(2).strip()
    else:
        parts = text.split(None, 1)
        headword = parts[0] if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

    equivalents = rest.lstrip("=: ").strip()
    return {
        "letter": letter,
        "headword": headword,
        "cross_ref": "; ".join(cross_refs),
        "equivalents": equivalents,
        "raw_entry": text,
    }


def format_entry_html(entry: str) -> str:
    """Normalize punctuation and apply semantic HTML formatting to a single entry."""
    text = normalize_entry_text(entry)
    escaped = html.escape(text)

    # Italicize cross-references: (Bak: ...) or (Bak ...)
    escaped = re.sub(
        r"\(Bak\s*:?\s*([^)]+)\)",
        lambda m: f"(Bak: <em>{m.group(1).strip()}</em>)",
        escaped,
    )

    # Bold the Turkish headword(s) up to the earliest delimiter (=, (Bak:, 1., or :)
    m = re.match(r"^(.+?)(\s*(?:=|\(Bak:|:\s+|\s+\d+\.\s+).*)$", escaped)
    if m:
        head, rest = m.group(1).strip(), m.group(2)
        return f'<p class="entry"><strong>{head}</strong>{rest}</p>'

    m = re.match(r"^(\S+)(\s+.*)?$", escaped)
    if m:
        head = m.group(1)
        rest = m.group(2) or ""
        return f'<p class="entry"><strong>{head}</strong>{rest}</p>'

    return f'<p class="entry">{escaped}</p>'


def extract_letter_entries(
    letter: str,
    start_page: int,
    end_page: int,
    page_texts: dict[int, str],
) -> list[str]:
    """Merge all pages of a dictionary letter into cleaned, de-hyphenated entries."""
    stream: list[str] = []
    for idx in range(start_page, end_page + 1):
        if idx in page_texts:
            stream.extend(clean_dictionary_page_lines(page_texts[idx]))

    entries: list[str] = []
    for ln in stream:
        if not entries or not is_continuation_line(ln, entries[-1], letter):
            entries.append(ln)
        else:
            entries[-1] = append_dehyphenated(entries[-1], ln)

    return [e for e in entries if e.strip()]


def build_letter_chapter_html(
    letter: str,
    start_page: int,
    end_page: int,
    page_texts: dict[int, str],
) -> tuple[str, int]:
    """Build seamless HTML for a single dictionary letter chapter."""
    entries = extract_letter_entries(letter, start_page, end_page, page_texts)
    entry_htmls = [format_entry_html(e) for e in entries]
    heading = f'<h1 class="letter-heading">{html.escape(letter)}</h1>'
    body = heading + "\n" + "\n".join(entry_htmls)
    return body, len(entry_htmls)


def export_dataset(page_texts: dict[int, str], export_dir: Path, stem: str) -> int:
    """Export structured dictionary entries as JSON and CSV datasets."""
    export_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for letter, start_p, end_p in DICTIONARY_LETTERS:
        if not any(p in page_texts for p in range(start_p, end_p + 1)):
            continue
        for entry in extract_letter_entries(letter, start_p, end_p, page_texts):
            records.append(parse_entry_fields(letter, entry))

    json_path = export_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = export_dir / f"{stem}.csv"
    fieldnames = ["letter", "headword", "cross_ref", "equivalents", "raw_entry"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return len(records)


def make_chapter(
    title: str,
    file_name: str,
    body_html: str,
    css: epub.EpubItem,
) -> epub.EpubHtml:
    """Create an EpubHtml chapter linked to the shared stylesheet."""
    chap = epub.EpubHtml(title=title, file_name=file_name, lang="tr")
    chap.content = (
        f"<html><head><title>{html.escape(title)}</title></head>"
        f"<body>\n{body_html}\n</body></html>"
    ).encode("utf-8")
    chap.add_item(css)
    return chap


def build_epub(
    page_texts: dict[int, str],
    cover_png: bytes | None,
    output_path: Path,
) -> None:
    """Assemble OCR'd pages into a polished, seamless A–Z EPUB."""
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
    margin-top: 2.5em;
}
.title-page .org {
    letter-spacing: 0.15em;
    font-weight: bold;
    margin-bottom: 2em;
}
.title-page .imprint {
    margin-top: 3em;
    line-height: 1.6;
}
.title-page .copyright {
    margin-top: 2.5em;
    font-size: 0.9em;
    font-style: italic;
}
p.prose {
    text-indent: 1.2em;
    margin: 0.45em 0;
    text-align: justify;
}
p.signature {
    text-align: right;
    font-style: italic;
    margin-top: 1.2em;
}
p.rule {
    margin: 0.4em 0;
    padding-left: 1.4em;
    text-indent: -1.4em;
}
p.subrule {
    margin: 0.3em 0;
    padding-left: 2.6em;
    text-indent: -1.2em;
}
p.entry {
    margin: 0.3em 0;
    padding-left: 1.4em;
    text-indent: -1.4em;
}
.colophon {
    text-align: center;
    margin-top: 2em;
}
.colophon p {
    margin: 0.3em 0;
}
""".strip().encode("utf-8"),
    )
    book.add_item(css)

    chapters: list[epub.EpubHtml] = []

    # Front matter (if pages are present)
    if 0 in page_texts or 1 in page_texts:
        chapters.append(make_chapter("Kapak / Künye", "00_kunye.xhtml", build_title_html(), css))
    if 2 in page_texts:
        chapters.append(make_chapter("Önsöz", "01_onsoz.xhtml", build_onsoz_html(page_texts), css))
    if any(i in page_texts for i in range(3, 8)):
        chapters.append(
            make_chapter(
                "Türk Dil Kurumunun Bildiriği",
                "02_bildirig.xhtml",
                build_bildirig_html(page_texts),
                css,
            )
        )
    if 8 in page_texts or 9 in page_texts:
        chapters.append(
            make_chapter("İmler ve Kısaltmalar", "03_imler.xhtml", build_imler_html(page_texts), css)
        )

    # Alphabetical dictionary chapters (A-Z)
    ascii_slug = {"Ç": "ch", "İ": "ii", "Ö": "oe", "Ş": "sh", "Ü": "ue"}
    for idx, (letter, start_p, end_p) in enumerate(DICTIONARY_LETTERS, start=1):
        if not any(p in page_texts for p in range(start_p, end_p + 1)):
            continue
        body_html, _ = build_letter_chapter_html(letter, start_p, end_p, page_texts)
        slug = ascii_slug.get(letter, letter.lower())
        chapters.append(
            make_chapter(letter, f"dict_{idx:02d}_{slug}.xhtml", body_html, css)
        )

    # Back matter
    if 352 in page_texts:
        chapters.append(
            make_chapter("Satış Merkezi", "99_satis_merkezi.xhtml", build_colophon_html(page_texts), css)
        )

    for chap in chapters:
        book.add_item(chap)

    book.toc = tuple(chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters]

    epub.write_epub(str(output_path), book)


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
    parser = argparse.ArgumentParser(description="Convert scanned PDF to EPUB via Cloud Vision API.")
    parser.add_argument("--pdf", help="Input PDF path (defaults to first PDF in cwd).")
    parser.add_argument("--output", help="Output EPUB path.")
    parser.add_argument("--project", help="GCP project ID (defaults to GOOGLE_CLOUD_PROJECT or gcloud config).")
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200).")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per Vision API request (max 16).")
    parser.add_argument("--pages", help="0-based page indices/ranges (e.g. 0,2,20-25).")
    parser.add_argument("--export-dir", help="Optional directory to also export JSON and CSV datasets.")
    args = parser.parse_args()

    pdf_path = find_pdf(args.pdf)
    output_path = Path(args.output) if args.output else pdf_path.with_suffix(".epub")
    cache_dir = Path(CACHE_DIR_NAME)
    cache_dir.mkdir(exist_ok=True)

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    target_pages = parse_page_selection(args.pages, total_pages)
    print(f"PDF: {pdf_path.name} ({total_pages} pages). Processing {len(target_pages)} page(s)...", flush=True)

    cover_png = doc[0].get_pixmap(dpi=args.dpi).tobytes("png") if total_pages > 0 else None

    page_texts: dict[int, str] = {}
    uncached: list[int] = []
    for idx in target_pages:
        cache_file = cache_dir / f"page_{idx:04d}.json"
        if cache_file.exists():
            page_texts[idx] = json.loads(cache_file.read_text(encoding="utf-8"))["text"]
        else:
            uncached.append(idx)

    if uncached:
        project_id = resolve_project_id(args.project)
        token = get_access_token()
        for start in range(0, len(uncached), args.batch_size):
            batch_indices = uncached[start : start + args.batch_size]
            batch_items = [
                (idx, doc[idx].get_pixmap(dpi=args.dpi).tobytes("png"))
                for idx in batch_indices
            ]
            results = ocr_batch(token, project_id, batch_items)
            for idx, text in results.items():
                page_texts[idx] = text
                (cache_dir / f"page_{idx:04d}.json").write_text(
                    json.dumps({"page": idx, "text": text}, ensure_ascii=False),
                    encoding="utf-8",
                )
            done = len(page_texts)
            print(f"Progress: {done}/{len(target_pages)} pages", flush=True)
    else:
        print("All pages loaded from .vision_cache (0 API calls).", flush=True)

    build_epub(page_texts, cover_png, output_path)
    print(f"Saved EPUB: {output_path}", flush=True)

    if args.export_dir:
        count = export_dataset(
            page_texts,
            Path(args.export_dir),
            "turkceden_osmanlicaya_cep_kilavuzu_1935",
        )
        print(f"Exported {count} entries to {args.export_dir}/", flush=True)


if __name__ == "__main__":
    main()

# Historical Turkish Book Digitizer

An automated Python pipeline to digitize public-domain historical Turkish books from scanned PDFs into clean, reflowable EPUBs and structured JSON/CSV lexicon datasets using Google Cloud Vision OCR (`DOCUMENT_TEXT_DETECTION`).

## Digitized Catalog

| Title | Author | Year | Pages | Entries | Dataset | Archive / Binary | Notes |
|---|---|---:|---:|---:|---|---|---|
| *Osmanlıcadan Türkçeye Söz Karşılıkları: Tarama Dergisi (Cilt 2)* | Türk Dili Tetkik Cemiyeti (T.D.T.C.) | 1934 | 459 | 29,615 | [JSON](data/tarama_dergisi_cilt2_1934.json) / [CSV](data/tarama_dergisi_cilt2_1934.csv) | TBD (`archive.org` / Releases) | Unproofread 2-column Cloud Vision OCR (*Türkçeden Osmanlıcaya İndeks*) |
| *Türkçeden Osmanlıcaya Cep Kılavuzu* | Türk Dil Kurumu (T.D.K.) | 1935 | 353 | 7,611 | [JSON](data/turkceden_osmanlicaya_cep_kilavuzu_1935.json) / [CSV](data/turkceden_osmanlicaya_cep_kilavuzu_1935.csv) | TBD (`archive.org` / Releases) | Unproofread Cloud Vision OCR; mixed Turkish, Ottoman Turkish, and French equivalents |

See [catalog.csv](catalog.csv) for machine-readable metadata.

## Features

- **Batched Cloud Vision OCR (`vision_to_epub.py`):** Renders PDF pages via `PyMuPDF` and sends up to 16 pages per `DOCUMENT_TEXT_DETECTION` request using `gcloud` OAuth2 credentials.
- **Resumable local cache:** Stores per-page OCR output in `.vision_cache/` so formatting updates and dataset exports rebuild in seconds with zero extra API calls.
- **Artifact removal & seamless reflow:**
  - Strips running 3-letter guide-words (e.g., `Alf`, `Ace`), Arabic/Roman page numbers, and stray OCR noise.
  - Merges wrapped lines and hyphenated words across line and page boundaries while preserving prefix/suffix markers (e.g., `Hem-`, `-i`).
  - Organizes books into clean front matter, `A`–`Z` alphabetical chapters, and colophon with hanging-indent typography and italicized `(Bak: ...)` cross-references.
- **Structured dataset export:** `--export-dir data` outputs normalized JSON and UTF-8 CSV records (`letter`, `headword`, `cross_ref`, `equivalents`, `raw_entry`).
- **Alternative Gemini API backend (`convert_to_epub.py`):** Includes a rate-limited (`--rpm`) Gemini Vision script for multimodal page transcription.

## Setup

Requires **Python 3.10+** and the **Google Cloud SDK (`gcloud`)** with Cloud Vision API enabled.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Authenticate with Google Cloud and set your project:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Usage

Run OCR (or rebuild from `.vision_cache/`), generate the `.epub`, and export structured JSON/CSV datasets:

```powershell
.\.venv\Scripts\python.exe -u vision_to_epub.py --pdf "Türkçeden Osmanlıcaya Cep Kılavuzu.pdf" --export-dir data
```

Optional flags:
- `--project <ID>`: Override GCP project ID (or set `GOOGLE_CLOUD_PROJECT`)
- `--dpi 200`: PDF rendering resolution
- `--batch-size 16`: Pages per Cloud Vision API batch request (max 16)
- `--pages 0,2,20-25`: Process a subset of 0-based page indices

## Disclaimer

Automated OCR conversions are unproofread (~98%+ character accuracy on clean 1935 Latin print, with occasional reading-order or diacritic artifacts). Keep the original scanned PDF as ground truth for scholarly citation.

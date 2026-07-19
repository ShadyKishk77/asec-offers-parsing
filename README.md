# ASEC Offers & Invoices Parsing Pipeline

An automated, GPU-accelerated pipeline to extract structured data from vendor offer letters, quotes, and invoices using a **local LLM (Ollama / Llama 3.1)**, validate and enrich the results, and export them to a professionally formatted Excel spreadsheet.

Supports digital PDFs and Arabic/English scanned documents (via Tesseract OCR fallback), with a Streamlit web dashboard for interactive review and editing.

---

## Key Features

- 🤖 **Local LLM via Ollama** — Uses `llama3.1-gpu` (Llama 3.1 8B) running locally. No cloud API required.
- 📄 **Table-Aware Extraction** — Extracts pdfplumber tables as Markdown grids for richer LLM context.
- 🔁 **Parallel Processing** — Processes multiple PDFs simultaneously via `ThreadPoolExecutor`.
- 🌍 **Dual-Language OCR** — Tesseract fallback with automatic visual Arabic (RTL) correction.
- 💱 **EGP & USD Currency Detection** — Auto-detects and formats Egyptian Pound and US Dollar.
- 🎯 **Confidence Scoring** — LLM self-reports confidence (0-100); rows below 80% auto-flagged.
- ✏️ **Interactive Editing** — Edit flagged rows in `st.data_editor` before downloading Excel.
- 💾 **Session State** — Results persist across sidebar configuration changes.

---

## Architecture

```
Input PDFs → Stage 1: Extractor (pdfplumber + tables + OCR)
           → Stage 2: LLM Client (Ollama llama3.1-gpu, few-shot prompting)
           → Stage 3: Validator (confidence flags + OCR name cleaning)
           → Stage 4: Exporter (openpyxl, EGP/USD formatting, Excel formulas)
           → Two-Sheet Excel File
```

---

## How to Set Up & Run

### 1. Prerequisites

1. **Ollama**: [https://ollama.com/download](https://ollama.com/download)
   ```bash
   ollama pull llama3.1:8b
   ollama create llama3.1-gpu -f Modelfile
   ollama serve
   ```
2. **Tesseract-OCR**: [Download for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
3. **Poppler**: [Download for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
4. **Python dependencies**: `pip install -r requirements.txt`

### 2. Configuration (`.env`)

```ini
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1-gpu

TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH=C:\poppler\Library\bin

# Optional multi-language OCR (Arabic + English)
TESSERACT_LANG=eng+ara
```

### 3. Run

```bash
# Web dashboard (recommended)
streamlit run app.py

# CLI batch processing
python pipeline.py --input-dir documents-sample
python pipeline.py --input-dir documents-sample --verbose
```

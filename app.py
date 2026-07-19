import os
import io
import shutil
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import streamlit as st

# Force set utf-8 encoding for outputs
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.extractor import extract_text_from_pdf
from src.llm_client import extract_document_data
from src.validator import validate_and_enrich
from src.exporter import export_to_excel
from src.schema import FlatRow

# Setup basic configuration
st.set_page_config(
    page_title="ASEC Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f0f4f8;
    }

    /* Page header banner */
    .page-header {
        background: linear-gradient(135deg, #1a2f5a 0%, #2d5086 100%);
        border-radius: 14px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        color: white;
    }
    .page-header h1 {
        color: white !important;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.3px;
    }
    .page-header p {
        color: rgba(255,255,255,0.75);
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }

    /* Primary action button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1a2f5a, #2d5086);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        letter-spacing: 0.2px;
        transition: all 0.25s ease;
        box-shadow: 0 2px 8px rgba(26, 47, 90, 0.25);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #2d5086, #3a64a8);
        transform: translateY(-1px);
        box-shadow: 0 5px 16px rgba(26, 47, 90, 0.3);
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-top: 3px solid #1a2f5a;
        text-align: center;
    }
    .metric-card.warn {
        border-top: 3px solid #e6a817;
    }
    .metric-card.good {
        border-top: 3px solid #2e7d4f;
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #7a8599;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a2f5a;
        line-height: 1;
    }
    .metric-value.warn { color: #c47f00; }
    .metric-value.good { color: #2e7d4f; }

    /* Status cards */
    .status-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        border-left: 4px solid #1a2f5a;
    }
    .status-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 0.92rem;
        font-weight: 600;
        color: #1a2f5a;
    }
    .status-card p {
        margin: 0;
        font-size: 0.83rem;
        color: #6b7280;
    }
    .status-card.flagged {
        border-left-color: #e6a817;
        background: #fffbf0;
    }
    .status-card.error {
        border-left-color: #d9534f;
        background: #fff5f5;
    }
    .status-card.flagged h4 { color: #a06c00; }
    .status-card.error h4 { color: #b94040; }

    /* Section divider */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9ca3af;
        margin-bottom: 0.8rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a2f5a;
    }
    section[data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stMarkdown hr {
        border-color: rgba(255,255,255,0.15);
    }
    section[data-testid="stSidebar"] .stInfo {
        background: rgba(255,255,255,0.1);
        border: none;
        border-radius: 8px;
    }

    h1, h2, h3 {
        color: #1a2f5a !important;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Page Header Banner
st.markdown("""
<div class="page-header">
    <h1>ASEC Document Intelligence</h1>
    <p>Upload vendor quotes and invoices — we extract, structure, and export the data automatically.</p>
</div>
""", unsafe_allow_html=True)

# Configure Temporary Folder
TEMP_DIR = Path(__file__).parent / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

# --- Session State Initialization ---
if "all_rows" not in st.session_state:
    st.session_state.all_rows = []
if "doc_summaries" not in st.session_state:
    st.session_state.doc_summaries = []
if "run_dir" not in st.session_state:
    st.session_state.run_dir = None
if "extraction_done" not in st.session_state:
    st.session_state.extraction_done = False

# Sidebar configurations
with st.sidebar:
    st.markdown("## ASEC Document Intelligence")
    st.markdown("---")
    st.markdown("### Capabilities")
    st.markdown("- AI-powered data extraction")
    st.markdown("- Scanned document support")
    st.markdown("- Arabic & English documents")
    st.markdown("- EGP & USD currency detection")
    st.markdown("- Confidence-based quality review")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("1. Upload your PDF documents below.")
    st.markdown("2. Click **Run Extraction**.")
    st.markdown("3. Review and correct any flagged rows.")
    st.markdown("4. Download the structured Excel report.")
    st.markdown("---")
    st.caption("Processes multiple files simultaneously.")

# File Uploader
uploaded_files = st.file_uploader(
    "Drag and drop PDF documents here", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} document(s) ready for processing.")

    if st.button("Run Extraction"):
        # Reset session state for a fresh run
        st.session_state.all_rows = []
        st.session_state.doc_summaries = []
        st.session_state.extraction_done = False

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        status_container = st.container()
        
        # Create unique directory for this run's uploads
        run_id = str(int(time.time()))
        run_dir = TEMP_DIR / run_id
        run_dir.mkdir(exist_ok=True)
        st.session_state.run_dir = str(run_dir)

        # Save all uploaded files to temp dir first
        temp_paths: dict[str, Path] = {}
        for uploaded_file in uploaded_files:
            temp_file_path = run_dir / uploaded_file.name
            with open(temp_file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file, f)
            temp_paths[uploaded_file.name] = temp_file_path

        total_docs = len(uploaded_files)
        completed_count = 0

        def _process_document(filename: str, temp_path: Path) -> dict:
            """Process a single document through all pipeline stages."""
            t0 = time.time()
            try:
                pages = extract_text_from_pdf(temp_path)
                doc = extract_document_data(pages, filename)
                ocr_failed_pages = {p.page_num for p in pages if p.ocr_failed}
                rows = validate_and_enrich(doc, filename, ocr_failed_pages or None)
                flagged = sum(1 for r in rows if r.needs_review)
                elapsed = time.time() - t0
                return {
                    "status": "ok",
                    "filename": filename,
                    "rows": rows,
                    "doc": doc,
                    "flagged": flagged,
                    "elapsed": elapsed,
                }
            except Exception as e:
                elapsed = time.time() - t0
                return {
                    "status": "error",
                    "filename": filename,
                    "error": str(e),
                    "elapsed": elapsed,
                }

        # --- Parallel Processing ---
        with ThreadPoolExecutor(max_workers=min(4, total_docs)) as executor:
            future_map = {
                executor.submit(_process_document, name, path): name
                for name, path in temp_paths.items()
            }
            for future in as_completed(future_map):
                result = future.result()
                completed_count += 1
                progress_bar.progress(completed_count / total_docs)
                status_text.markdown(f"**Processed {completed_count}/{total_docs}:** `{result['filename']}`")

                if result["status"] == "ok":
                    st.session_state.all_rows.extend(result["rows"])
                    review_status = "Needs Review" if result["flagged"] > 0 else "Complete"
                    st.session_state.doc_summaries.append({
                        "Document": result["filename"],
                        "Vendor": result["doc"].company_name,
                        "Date": result["doc"].date,
                        "Currency": result["doc"].currency,
                        "Line Items": len(result["rows"]),
                        "Flagged": result["flagged"],
                        "Status": review_status,
                        "Time (s)": round(result["elapsed"], 1),
                    })
                    card_class = "flagged" if result["flagged"] > 0 else ""
                    with status_container:
                        st.markdown(f"""
                        <div class="status-card {card_class}">
                            <h4>{result['filename']}</h4>
                            <p>Vendor: {result['doc'].company_name} &nbsp;|&nbsp; Date: {result['doc'].date} &nbsp;|&nbsp; {len(result['rows'])} line item(s) extracted &nbsp;|&nbsp; {result['flagged']} flagged &nbsp;|&nbsp; {result['elapsed']:.1f}s</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.session_state.doc_summaries.append({
                        "Document": result["filename"],
                        "Vendor": "—",
                        "Date": "—",
                        "Currency": "—",
                        "Line Items": 0,
                        "Flagged": 0,
                        "Status": "Failed",
                        "Time (s)": round(result["elapsed"], 1),
                    })
                    with status_container:
                        st.markdown(f"""
                        <div class="status-card error">
                            <h4>{result['filename']}</h4>
                            <p>Could not process this document. Please check the file and try again.</p>
                        </div>
                        """, unsafe_allow_html=True)

        status_text.markdown("**All documents processed.** Review the results below.")
        st.session_state.extraction_done = True


# --- Dashboard (persists across sidebar changes via session_state) ---
if st.session_state.extraction_done and st.session_state.all_rows:
    all_rows = st.session_state.all_rows
    doc_summaries = st.session_state.doc_summaries

    st.write("---")
    st.markdown('<p class="section-label">Summary</p>', unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    total_cost = sum(r.line_total for r in all_rows)
    total_flagged = sum(1 for r in all_rows if r.needs_review)
    
    m_col1.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Documents</div>
        <div class="metric-value">{len(doc_summaries)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    m_col2.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Line Items Extracted</div>
        <div class="metric-value">{len(all_rows)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    m_col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Quoted Value</div>
        <div class="metric-value" style="font-size:1.4rem">{total_cost:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    flagged_class = "warn" if total_flagged > 0 else "good"
    value_class = "warn" if total_flagged > 0 else "good"
    m_col4.markdown(f"""
    <div class="metric-card {flagged_class}">
        <div class="metric-label">Require Review</div>
        <div class="metric-value {value_class}">{total_flagged}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.markdown('<p class="section-label">Extracted Data</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Line Items", "Document Summary"])
    
    with tab1:
        st.caption("Review and edit the extracted data below. Any changes you make will be reflected in the downloaded report.")

        def _confidence_label(conf) -> str:
            if conf is None:
                return "—"
            if conf >= 80:
                return f"High ({conf}%)"
            elif conf >= 60:
                return f"Medium ({conf}%)"
            else:
                return f"Low ({conf}%)"

        items_data = []
        for r in all_rows:
            items_data.append({
                "Source File": r.source_file,
                "Vendor Name": r.company_name,
                "Date": r.date,
                "SKU": r.sku or "",
                "Item Name": r.item_name,
                "Currency": r.currency,
                "Price": r.price,
                "Quantity": r.quantity,
                "Tax": r.tax,
                "Total (PDF)": r.total_amount,
                "Line Total": r.line_total,
                "Confidence": _confidence_label(r.confidence),
                "Status": "Needs Review" if r.needs_review else "OK",
                "Review Notes": r.review_reason or "",
            })
        
        df_items = pd.DataFrame(items_data)

        edited_df = st.data_editor(
            df_items,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Source File": st.column_config.TextColumn("Source File", disabled=True),
                "Vendor Name": st.column_config.TextColumn("Vendor"),
                "Date": st.column_config.TextColumn("Date"),
                "SKU": st.column_config.TextColumn("Part No."),
                "Item Name": st.column_config.TextColumn("Item Description"),
                "Currency": st.column_config.SelectboxColumn("Currency", options=["EGP", "USD"]),
                "Price": st.column_config.NumberColumn("Unit Price", format="%.2f"),
                "Quantity": st.column_config.NumberColumn("Qty", format="%.2f"),
                "Tax": st.column_config.NumberColumn("Tax", format="%.2f"),
                "Total (PDF)": st.column_config.NumberColumn("Total (Document)", format="%.2f", disabled=True),
                "Line Total": st.column_config.NumberColumn("Calculated Total", format="%.2f", disabled=True),
                "Confidence": st.column_config.TextColumn("AI Confidence", disabled=True),
                "Status": st.column_config.TextColumn("Status", disabled=True),
                "Review Notes": st.column_config.TextColumn("Review Notes", disabled=True),
            },
        )
    
    with tab2:
        st.caption("One row per processed document.")
        df_docs = pd.DataFrame(doc_summaries)
        st.dataframe(df_docs, use_container_width=True)
    
    st.write("")
    st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)

    # Rebuild FlatRows from the edited dataframe (captures user corrections)
    edited_rows: list[FlatRow] = []
    for orig_row, (_, ed_row) in zip(all_rows, edited_df.iterrows()):
        try:
            edited_rows.append(FlatRow(
                source_file=orig_row.source_file,
                company_name=str(ed_row["Vendor Name"]).strip() or orig_row.company_name,
                date=str(ed_row["Date"]).strip() or orig_row.date,
                currency=str(ed_row["Currency"]).strip() or orig_row.currency,
                sku=str(ed_row["SKU"]).strip() or None,
                item_name=str(ed_row["Item Name"]).strip() or orig_row.item_name,
                description=orig_row.description,
                price=float(ed_row["Price"]),
                quantity=float(ed_row["Quantity"]),
                tax=float(ed_row["Tax"]),
                total_amount=orig_row.total_amount,
                line_total=round((float(ed_row["Price"]) * float(ed_row["Quantity"])) + float(ed_row["Tax"]), 6),
                needs_review=orig_row.needs_review,
                review_reason=orig_row.review_reason,
                confidence=orig_row.confidence,
            ))
        except Exception:
            edited_rows.append(orig_row)

    # Generate Excel binary buffer in-memory
    run_dir_path = Path(st.session_state.run_dir) if st.session_state.run_dir else TEMP_DIR
    run_dir_path.mkdir(exist_ok=True)
    excel_out_path = run_dir_path / "extraction_output.xlsx"
    export_to_excel(edited_rows, excel_out_path)
    
    with open(excel_out_path, "rb") as f:
        excel_data = f.read()
    
    st.download_button(
        label="Download Excel Report",
        data=excel_data,
        file_name="asec_extraction_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

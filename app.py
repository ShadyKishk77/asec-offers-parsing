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
# Setup basic configuration
st.set_page_config(
    page_title="ASEC Document Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Navy Blue Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F4F6F9;
    }

    /* Enterprise Navy Header */
    .hero-container {
        background: linear-gradient(135deg, #0B192C 0%, #1E3E62 60%, #000000 100%);
        border-radius: 16px;
        padding: 3rem 3.2rem;
        margin-bottom: 2rem;
        color: #FFFFFF;
        box-shadow: 0 20px 25px -5px rgba(11, 25, 44, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        top: -40%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(56, 122, 223, 0.2) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 122, 223, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(147, 197, 253, 0.3);
        padding: 0.35rem 0.95rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        margin-bottom: 1.1rem;
        text-transform: uppercase;
    }
    .hero-container h1 {
        font-family: 'Outfit', sans-serif !important;
        color: #FFFFFF !important;
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
        letter-spacing: -0.5px;
        line-height: 1.25;
    }
    .hero-container p {
        color: #CBD5E1;
        font-size: 1.05rem;
        margin: 0 0 1.6rem 0;
        font-weight: 400;
        max-width: 780px;
        line-height: 1.6;
    }
    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #E2E8F0;
        padding: 0.4rem 0.95rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Feature Showcase Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1.2rem;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.35rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-top: 3px solid #1E3E62;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        border-top-color: #008DDA;
    }
    .feature-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #0B192C !important;
        margin: 0 0 0.35rem 0;
    }
    .feature-card p {
        font-size: 0.85rem;
        color: #64748B;
        margin: 0;
        line-height: 1.45;
    }

    /* Primary Navy Action Buttons */
    div.stButton > button:first-child {
        background: #1E3E62;
        color: white;
        border-radius: 8px;
        padding: 0.65rem 2.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        letter-spacing: 0.2px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(30, 62, 98, 0.25);
    }
    div.stButton > button:first-child:hover {
        background: #0B192C;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(11, 25, 44, 0.35);
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        border: 1px solid #E2E8F0;
        border-top: 4px solid #1E3E62;
        text-align: center;
    }
    .metric-card.warn {
        border-top: 4px solid #D97706;
    }
    .metric-card.good {
        border-top: 4px solid #059669;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #0B192C;
        line-height: 1;
    }
    .metric-value.warn { color: #D97706; }
    .metric-value.good { color: #059669; }

    /* Status Cards */
    .status-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        border: 1px solid #E2E8F0;
        border-left: 4px solid #059669;
    }
    .status-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #0B192C;
    }
    .status-card p {
        margin: 0;
        font-size: 0.84rem;
        color: #64748B;
    }
    .status-card.flagged {
        border-left-color: #D97706;
        background: #FFFBEB;
    }
    .status-card.error {
        border-left-color: #DC2626;
        background: #FEF2F2;
    }

    /* Section Label */
    .section-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        color: #475569;
        margin-bottom: 0.85rem;
    }

    /* Deep Navy Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0B192C;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-family: 'Outfit', sans-serif;
    }
    section[data-testid="stSidebar"] .stMarkdown hr {
        border-color: #1E293B;
    }

    h1, h2, h3 {
        color: #0B192C !important;
        font-family: 'Outfit', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.92rem;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# Executive Front Landing Page Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Document Intelligence Platform</div>
    <h1>Automated Vendor Quote & Invoice Extraction System</h1>
    <p>Upload vendor quotes, invoices, and tender letters in Arabic and English. Automatically extract, structure, and export line items, prices, 14% VAT, payment terms, and delivery schedules.</p>
    <div class="hero-pills">
        <span class="hero-pill">Multi-Language OCR</span>
        <span class="hero-pill">14% VAT Reconciliation</span>
        <span class="hero-pill">Payment & Delivery Parsing</span>
        <span class="hero-pill">Excel Export</span>
        <span class="hero-pill">Document Viewer</span>
    </div>
</div>

<div class="feature-grid">
    <div class="feature-card">
        <h3>Automated Line Extraction</h3>
        <p>Extracts SKUs, item descriptions, quantities, unit prices, and currencies without manual data entry.</p>
    </div>
    <div class="feature-card">
        <h3>Smart VAT & Math Check</h3>
        <p>Detects 14% inclusive/exclusive VAT in Terms & Conditions and reconciles line totals.</p>
    </div>
    <div class="feature-card">
        <h3>Payment & Delivery Parsing</h3>
        <p>Identifies payment terms, deferred credit windows, and lead time availability schedules.</p>
    </div>
    <div class="feature-card">
        <h3>Document Viewer</h3>
        <p>Inspect original source documents page-by-page directly alongside extracted data tables.</p>
    </div>
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
    st.markdown("## System Configuration")
    st.markdown("---")
    
    from src.llm_client import _OLLAMA_HOST, _MODEL_NAME, _OR_API_KEY, _OR_MODEL
    
    # API Key is read securely from environment / secrets in the background
    ui_or_key = _OR_API_KEY
    
    available_or_models = [
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-flash-1.5",
        "openai/gpt-4o-mini",
        "deepseek/deepseek-chat",
        "qwen/qwen-2.5-72b-instruct",
    ]
    
    default_idx = 0
    if _OR_MODEL in available_or_models:
        default_idx = available_or_models.index(_OR_MODEL)

    ui_or_model = st.selectbox(
        "AI Engine Model",
        options=available_or_models,
        index=default_idx,
        help="Select the AI model engine for document processing.",
    )
    
    active_or_key = (ui_or_key or "").strip()
    if active_or_key:
        st.info("Status: OpenRouter Cloud API (Connected)")
    else:
        st.info("Status: Local Engine (Ollama Active)")
    
    st.markdown("---")
    st.markdown("### System Features")
    st.markdown("• Multi-language PDF text extraction")
    st.markdown("• Scanned document OCR processing")
    st.markdown("• Automatic 14% VAT reconciliation")
    st.markdown("• EGP & USD currency parsing")
    st.markdown("• Two-sheet structured Excel export")
    st.markdown("---")
    st.markdown("### User Instructions")
    st.markdown("1. Upload PDF document files.")
    st.markdown("2. Click **Process Documents**.")
    st.markdown("3. Review extracted data tables.")
    st.markdown("4. Download the Excel report.")
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
                doc = extract_document_data(
                    pages, 
                    filename, 
                    api_key_override=ui_or_key, 
                    model_override=ui_or_model
                )
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
                logging.exception(f"Error processing document: {filename}")
                elapsed = time.time() - t0
                return {
                    "status": "error",
                    "filename": filename,
                    "error": f"{type(e).__name__}: {str(e)}",
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
                        "Payment Terms": result["doc"].payment_terms or "—",
                        "Delivery Time": result["doc"].delivery_time or "—",
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
                            <p>Could not process this document: <code>{result['error']}</code></p>
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
    st.write("")
    st.markdown('<p class="section-label">Extracted Data & Document Viewer</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Line Items", "Document Summary", "Document Preview"])

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
                "Payment Terms": r.payment_terms or "—",
                "Delivery Time": r.delivery_time or "—",
                "Validity": r.offer_validity or "—",
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

        # --- Interactive Filters ---
        f_col1, f_col2 = st.columns([2, 2])
        all_vendors = sorted(list(set(df_items["Vendor Name"].unique())))
        selected_vendors = f_col1.multiselect("Filter by Vendor", options=all_vendors, default=all_vendors)
        status_filter = f_col2.selectbox("Filter by Review Status", options=["All Rows", "Needs Review Only", "Clean Rows Only"])

        filtered_df = df_items[df_items["Vendor Name"].isin(selected_vendors)]
        if status_filter == "Needs Review Only":
            filtered_df = filtered_df[filtered_df["Status"] == "Needs Review"]
        elif status_filter == "Clean Rows Only":
            filtered_df = filtered_df[filtered_df["Status"] == "OK"]

        edited_df = st.data_editor(
            filtered_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Source File": st.column_config.TextColumn("Source File", disabled=True),
                "Vendor Name": st.column_config.TextColumn("Vendor"),
                "Date": st.column_config.TextColumn("Date"),
                "Payment Terms": st.column_config.TextColumn("Payment Policy"),
                "Delivery Time": st.column_config.TextColumn("Delivery Time"),
                "Validity": st.column_config.TextColumn("Validity"),
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

    with tab3:
        st.markdown("### Document Viewer")
        unique_files = sorted(list(set(df_items["Source File"].unique())))
        if unique_files:
            pv_col1, pv_col2 = st.columns([3, 1])
            selected_pdf = pv_col1.selectbox("Select Document to View", options=unique_files)
            run_dir_path = Path(st.session_state.run_dir) if st.session_state.run_dir else TEMP_DIR
            target_pdf_path = run_dir_path / selected_pdf
            
            if target_pdf_path.exists():
                import pdfplumber
                with open(target_pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                pv_col2.write("")
                pv_col2.write("")
                pv_col2.download_button("Download Document", data=pdf_bytes, file_name=selected_pdf, mime="application/pdf", use_container_width=True)
                
                try:
                    with pdfplumber.open(target_pdf_path) as pdf:
                        total_pages = len(pdf.pages)
                        if total_pages > 1:
                            selected_page_num = st.slider("Select Page", 1, total_pages, 1)
                        else:
                            selected_page_num = 1
                        
                        page = pdf.pages[selected_page_num - 1]
                        page_img = page.to_image(resolution=150).original
                        st.image(page_img, use_container_width=True, caption=f"Document: {selected_pdf} — Page {selected_page_num} of {total_pages}")
                except Exception as err:
                    st.error(f"Could not render document preview: {err}")
            else:
                st.warning(f"File '{selected_pdf}' is not available for preview.")
    
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

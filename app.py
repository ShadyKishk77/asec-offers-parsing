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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F8FAFC;
    }

    /* Page header banner with rich gradient & glow */
    .page-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%);
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .page-header::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .page-header h1 {
        font-family: 'Outfit', sans-serif !important;
        color: #FFFFFF !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .page-header p {
        color: #C7D2FE;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
        max-width: 650px;
        line-height: 1.5;
    }

    /* Primary action buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%);
        color: white;
        border-radius: 10px;
        padding: 0.65rem 2.4rem;
        font-weight: 600;
        font-size: 0.98rem;
        border: none;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
    }

    /* Metric glassmorphic cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        border: 1px solid #E2E8F0;
        border-top: 4px solid #4F46E5;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.07);
    }
    .metric-card.warn {
        border-top: 4px solid #F59E0B;
    }
    .metric-card.good {
        border-top: 4px solid #10B981;
    }
    .metric-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.6rem;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1;
    }
    .metric-value.warn { color: #D97706; }
    .metric-value.good { color: #059669; }

    /* Status cards */
    .status-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #E2E8F0;
        border-left: 5px solid #10B981;
        transition: transform 0.15s ease;
    }
    .status-card:hover {
        transform: translateX(3px);
    }
    .status-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 0.98rem;
        font-weight: 600;
        color: #0F172A;
    }
    .status-card p {
        margin: 0;
        font-size: 0.86rem;
        color: #64748B;
    }
    .status-card.flagged {
        border-left-color: #F59E0B;
        background: #FFFBEB;
    }
    .status-card.error {
        border-left-color: #EF4444;
        background: #FEF2F2;
    }
    .status-card.flagged h4 { color: #B45309; }
    .status-card.error h4 { color: #B91C1C; }

    /* Section label */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748B;
        margin-bottom: 0.9rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
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
        border-color: #334155;
    }
    section[data-testid="stSidebar"] .stInfo {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
    }

    h1, h2, h3 {
        color: #0F172A !important;
        font-family: 'Outfit', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569;
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
    
    # LLM Settings & Connection Diagnostics
    st.markdown("### LLM Provider Settings")
    from src.llm_client import _OLLAMA_HOST, _MODEL_NAME, _OR_API_KEY, _OR_MODEL
    
    ui_or_key = st.text_input(
        "OpenRouter API Key (optional)",
        value=_OR_API_KEY,
        type="password",
        help="If set, uses OpenRouter Cloud API instead of local Ollama.",
    )
    
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
        "OpenRouter Model",
        options=available_or_models,
        index=default_idx,
        help="Select model when using OpenRouter API.",
    )
    
    active_or_key = (ui_or_key or "").strip()
    if active_or_key:
        st.info("⚡ Using **OpenRouter Cloud API**")
        st.caption(f"**Model:** `{ui_or_model}`")
    else:
        st.info("💻 Using **Ollama (Local)**")
        st.caption(f"**Target Host:** `{_OLLAMA_HOST}`")
        st.caption(f"**Target Model:** `{_MODEL_NAME}`")
    
    if st.button("Test Connection"):
        import httpx
        if active_or_key:
            try:
                res = httpx.get(
                    "https://openrouter.ai/api/v1/auth/key",
                    headers={"Authorization": f"Bearer {active_or_key}"},
                    timeout=5.0
                )
                if res.status_code == 200:
                    st.success(f"OpenRouter API connected! Model: {ui_or_model}")
                else:
                    st.error(f"OpenRouter returned status {res.status_code}: {res.text}")
            except Exception as err:
                st.error(f"Failed to connect to OpenRouter: {err}")
        else:
            try:
                res = httpx.get(_OLLAMA_HOST, timeout=5.0)
                if res.status_code == 200 or "Ollama is running" in res.text:
                    st.success("Ollama is reachable!")
                else:
                    st.error(f"Ollama returned status: {res.status_code}")
            except Exception as err:
                st.error(f"Failed to connect to Ollama: {err}")
            
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
    tab1, tab2, tab3 = st.tabs(["Line Items", "Document Summary", "PDF Preview 📄"])

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
        st.markdown("### 📄 PDF Document Viewer")
        unique_files = sorted(list(set(df_items["Source File"].unique())))
        if unique_files:
            pv_col1, pv_col2 = st.columns([3, 1])
            selected_pdf = pv_col1.selectbox("Select PDF Document to View", options=unique_files)
            run_dir_path = Path(st.session_state.run_dir) if st.session_state.run_dir else TEMP_DIR
            target_pdf_path = run_dir_path / selected_pdf
            
            if target_pdf_path.exists():
                import pdfplumber
                with open(target_pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                pv_col2.write("")
                pv_col2.write("")
                pv_col2.download_button("📥 Download PDF", data=pdf_bytes, file_name=selected_pdf, mime="application/pdf", use_container_width=True)
                
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
                    st.error(f"Could not render PDF preview: {err}")
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

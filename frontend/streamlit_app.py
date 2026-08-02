"""Streamlit frontend for the Dual RAG System."""

import os
import sys
from pathlib import Path

import streamlit as st
import httpx

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

SRC_DIR = Path(__file__).resolve().parent / "src"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MAID - My Assistant Integrated Device",
    page_icon="🐱",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Pastel pink theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background-color: #FFF0F5;
    }
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #FFE4EC;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #D87093 !important;
    }
    /* Text */
    .stMarkdown, p, li, span {
        color: #8B5A7A;
    }
    /* Buttons */
    .stButton button {
        background-color: #FFB6C1;
        color: #6B3A5A;
        border: 1px solid #FF91A4;
    }
    .stButton button:hover {
        background-color: #FF91A4;
        color: #6B3A5A;
        border: 1px solid #FF6F91;
    }
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #FFF5F8;
        border: 1px solid #FFD6E0;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #FFF5F8;
        border: 1px solid #FFD6E0;
        border-radius: 8px;
        padding: 10px;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #FFE4EC;
        color: #D87093 !important;
    }
    /* Success/Info/Warning/Error boxes */
    .stAlert {
        background-color: #FFF5F8;
        border: 1px solid #FFD6E0;
    }
    /* Input fields */
    input, textarea {
        background-color: #FFF5F8 !important;
        border: 1px solid #FFD6E0 !important;
        color: #8B5A7A !important;
    }
    /* Divider */
    hr {
        border-color: #FFD6E0;
    }
    /* Code blocks */
    code {
        background-color: #FFE4EC !important;
        color: #8B5A7A !important;
    }
    /* Select box */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFF5F8;
        border-color: #FFD6E0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header: catmaid logo + MAID title
# ---------------------------------------------------------------------------
catmaid_path = SRC_DIR / "catmaid.png"
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image(str(catmaid_path), width=80)
with col_title:
    st.markdown(
        "<h1 style='margin-bottom: 0; padding-bottom: 0;'>MAID</h1>"
        "<p style='margin-top: -10px; font-size: 1.1rem; color: #B06A8A;'>"
        "My Assistant Integrated Device</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    "Query documents using **Vector RAG** (docling → Qdrant, semantic) "
    "and **BM25** (keyword-based, vectorless) simultaneously."
)


# ---------------------------------------------------------------------------
# Sidebar: Status & Ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # API key status
    groq_ok = settings.groq_api_key != "your-groq-api-key-here"

    st.markdown("**Groq API:** " + ("✅ Configured" if groq_ok else "❌ Not configured"))
    st.markdown("**BM25 (vectorless):** ✅ Local (no API key needed)")
    st.markdown(f"**Groq Model:** `{settings.groq_model}`")

    st.divider()

    # Character image
    personagem_path = SRC_DIR / "personagem_maid.png"
    st.image(str(personagem_path), width=100)

    # Ingestion
    st.header("📄 Ingest PDFs")
    ingest_tab = st.radio("Mode", ["Upload files", "Browse server folders"], horizontal=True)

    if ingest_tab == "Upload files":
        uploaded_files = st.file_uploader(
            "Choose PDFs (drag & drop or select multiple)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            num_files = len(uploaded_files)
            st.caption(f"{num_files} file(s) selected")
            tmp_dir = Path("/tmp/dual_rag_uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)

            if st.button("Ingest All", type="primary"):
                progress = st.progress(0, text="Starting...")
                status_area = st.empty()
                succeeded = 0
                failed = 0

                for i, uploaded_file in enumerate(uploaded_files):
                    progress.progress(
                        (i + 1) / num_files,
                        text=f"Ingesting {uploaded_file.name} ({i + 1}/{num_files})...",
                    )
                    # Save temporarily
                    tmp_path = tmp_dir / uploaded_file.name
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    try:
                        resp = httpx.post(
                            f"{API_BASE}/api/ingest",
                            params={"file_path": str(tmp_path)},
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            succeeded += 1
                        else:
                            failed += 1
                            if status_area:
                                status_area.warning(f"⚠️ {uploaded_file.name}: {resp.text}")
                    except httpx.ConnectError:
                        failed += 1
                        st.error("Cannot connect to the backend. Is the FastAPI server running?")
                        break

                progress.empty()
                if succeeded > 0:
                    st.success(f"✅ Ingested {succeeded} file(s) successfully")
                if failed > 0:
                    st.error(f"❌ {failed} file(s) failed")
                if succeeded > 0:
                    st.rerun()

    else:
        # Browse server folders – navigate the filesystem visually
        # Store the current browsing path in session state
        if "browse_path" not in st.session_state:
            st.session_state.browse_path = str(Path.home())

        current = Path(st.session_state.browse_path).resolve()

        # Navigation breadcrumb
        st.markdown(f"**📍 {current}**")

        # Parent directory button
        parent = current.parent
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("📂 ..", key="go_up", help="Go to parent folder"):
                st.session_state.browse_path = str(parent)
                st.rerun()
        with col2:
            # Count PDFs in current directory
            pdf_count = len(list(current.glob("*.pdf"))) + len(list(current.glob("*.PDF")))
            st.caption(f"{pdf_count} PDF(s) in this folder")

        # List subdirectories (sorted, dirs first)
        try:
            entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            st.error(f"Permission denied: {current}")
            entries = []

        # Display subdirectories as clickable buttons
        # Use columns for a compact layout: 3 per row
        dirs = [e for e in entries if e.is_dir()]
        cols_per_row = 3

        for i in range(0, len(dirs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(dirs):
                    d = dirs[idx]
                    # Count nested PDFs (shallow)
                    nested_pdfs = len(list(d.glob("*.pdf"))) + len(list(d.glob("*.PDF")))
                    label = f"📁 {d.name}"
                    if nested_pdfs:
                        label += f" ({nested_pdfs})"
                    with col:
                        if st.button(label, key=f"nav_{idx}", help=str(d), use_container_width=True):
                            st.session_state.browse_path = str(d)
                            st.rerun()

        if not dirs:
            st.info("No subdirectories here.")

        # Ingest button for the current folder
        st.divider()
        if pdf_count > 0:
            if st.button(f"📥 Ingest all {pdf_count} PDF(s) from this folder", type="primary"):
                pdfs = sorted(current.glob("*.pdf")) + sorted(current.glob("*.PDF"))
                progress = st.progress(0, text="Starting...")
                status_area = st.empty()
                succeeded = 0
                failed = 0
                total = len(pdfs)

                for i, pdf in enumerate(pdfs):
                    progress.progress(
                        (i + 1) / total,
                        text=f"Ingesting {pdf.name} ({i + 1}/{total})...",
                    )
                    try:
                        resp = httpx.post(
                            f"{API_BASE}/api/ingest",
                            params={"file_path": str(pdf)},
                            timeout=300,
                        )
                        if resp.status_code == 200:
                            succeeded += 1
                        else:
                            failed += 1
                            if status_area:
                                status_area.warning(f"⚠️ {pdf.name}: {resp.text}")
                    except httpx.ConnectError:
                        failed += 1
                        st.error("Backend unreachable.")
                        break

                progress.empty()
                if succeeded > 0:
                    st.success(f"✅ Ingested {succeeded} file(s) successfully")
                if failed > 0:
                    st.error(f"❌ {failed} file(s) failed")
                if succeeded > 0:
                    st.rerun()
        else:
            st.caption("No PDFs in this folder — navigate to a folder with PDFs.")

    st.divider()

    # Document list
    st.header("📚 Indexed Documents")
    if st.button("Refresh document list"):
        st.rerun()

    try:
        resp = httpx.get(f"{API_BASE}/api/documents", timeout=10)
        if resp.status_code == 200:
            docs = resp.json()
            if docs:
                for doc in docs:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        vr = "✅" if doc["vector_rag_indexed"] else "❌"
                        bm = "✅" if doc["bm25_indexed"] else "❌"
                        st.markdown(f"**{doc['filename']}**")
                        st.markdown(f"Vector: {vr} | BM25: {bm} | Chunks: {doc['chunks']}")
                    with col2:
                        delete_key = f"delete_{doc['filename']}"
                        if st.button("🗑️", key=delete_key, help=f"Delete {doc['filename']}"):
                            try:
                                del_resp = httpx.delete(
                                    f"{API_BASE}/api/documents/{doc['filename']}",
                                    timeout=30,
                                )
                                if del_resp.status_code == 200:
                                    st.success(f"Deleted {doc['filename']}")
                                    st.rerun()
                                elif del_resp.status_code == 403:
                                    st.warning("Deletion is disabled in configuration.")
                                else:
                                    st.error(f"Delete failed: {del_resp.text}")
                            except httpx.ConnectError:
                                st.error("Cannot connect to the backend.")
            else:
                st.info("No documents indexed yet.")
        else:
            st.warning("Could not fetch document list.")
    except httpx.ConnectError:
        st.warning("Backend not reachable.")


# ---------------------------------------------------------------------------
# Main: Chat interface
# ---------------------------------------------------------------------------
st.header("💬 Ask a Question")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Sources", expanded=False):
                for src in msg["sources"]:
                    method = src["method"]
                    icon = "🧠" if method == "vector_rag" else "🌳"
                    st.markdown(f"{icon} **{method}** — `{src['document']}`")
                    if src.get("page"):
                        st.markdown(f"   Page: {src['page']}")
                    if src.get("score") is not None:
                        st.markdown(f"   Score: {src['score']:.4f}")
                    st.markdown(f"   ```\n{src['content'][:300]}...\n```")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching both RAG systems..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/api/query",
                    json={"question": prompt, "top_k": 3},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 Sources", expanded=True):
                            for src in sources:
                                method = src["method"]
                                icon = "🧠" if method == "vector_rag" else "🌳"
                                st.markdown(f"{icon} **{method}** — `{src['document']}`")
                                if src.get("page"):
                                    st.markdown(f"   Page: {src['page']}")
                                if src.get("score") is not None:
                                    st.markdown(f"   Score: {src['score']:.4f}")
                                st.markdown(f"   ```\n{src['content'][:300]}...\n```")

                    # Store in session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    st.error(f"Error: {resp.text}")
            except httpx.ConnectError:
                st.error("Cannot connect to the backend. Is the FastAPI server running?")

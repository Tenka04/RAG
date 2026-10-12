"""
Streamlit Web UI for Local & Cloud RAG Application
"""

import os
import tempfile
import streamlit as st
from pathlib import Path

from rag_engine import RAGEngine

# Set Streamlit page configuration
st.set_page_config(
    page_title="Local & Cloud RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a modern, sleek interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #94A3B8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }

    .source-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #3B82F620;
        color: #60A5FA;
        border: 1px solid #3B82F640;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_engine():
    """Cache the RAG engine instance across app reruns."""
    return RAGEngine()


engine = get_rag_engine()

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings & Models")

    # Ollama Health Check
    stats = engine.get_stats()
    if stats["ollama_online"]:
        st.success("🟢 Ollama Server: **Online**")
    else:
        st.warning("🟡 Ollama Server: **Offline**\nRun `ollama serve` in a terminal.")

    # Model Configuration
    st.markdown("#### 🤖 LLM Generation Model")
    model_mode = st.radio(
        "Mode:",
        ["Local Models (Offline & Free)", "Cloud Models (:cloud)"],
        index=0
    )

    if model_mode == "Local Models (Offline & Free)":
        selected_model = st.selectbox(
            "Select Local Model:",
            ["llama3.2:3b", "llama3.2:1b", "custom"],
            index=0,
            help="llama3.2:3b is recommended for your RTX 3050 GPU (fits completely in 4GB VRAM)."
        )
        if selected_model == "custom":
            selected_model = st.text_input("Enter custom model tag:", value="llama3.2:3b")
    else:
        selected_model = st.selectbox(
            "Select Cloud Model:",
            ["gemma4:cloud", "custom_cloud"],
            index=0,
            help="Requires 'ollama signin'. Runs remotely in Ollama's cloud."
        )
        if selected_model == "custom_cloud":
            selected_model = st.text_input("Enter cloud model tag (ends with :cloud):", value="gemma4:cloud")

    top_k = st.slider("Context Chunks (Top-K):", min_value=1, max_value=8, value=3)

    st.markdown("---")
    st.markdown("### 📚 Document Library")
    
    st.metric("Total Indexed Chunks", stats["total_chunks"])

    if stats["unique_documents"]:
        st.markdown("**Indexed Files:**")
        for doc in stats["unique_documents"]:
            st.markdown(f"- 📄 `{doc}`")
    else:
        st.info("No documents indexed yet. Upload a file or load sample docs below.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Load Samples", use_container_width=True):
            sample_dir = Path("./sample_docs")
            if sample_dir.exists():
                count = 0
                for f in sample_dir.glob("*.*"):
                    count += engine.ingest_file(str(f))
                st.toast(f"Indexed {count} chunks from sample documents!", icon="✅")
                st.rerun()
            else:
                st.error("sample_docs/ directory not found.")

    with col2:
        if st.button("🗑️ Clear Index", use_container_width=True):
            engine.clear_index()
            st.toast("Knowledge base cleared!", icon="🧹")
            st.rerun()


# -------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------
st.markdown('<div class="main-title">🧠 Local & Cloud RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ask questions grounded in your private documents using Ollama & ChromaDB</div>', unsafe_allow_html=True)

# Document Ingestion Section
with st.expander("📤 Upload & Ingest New Documents", expanded=(stats["total_chunks"] == 0)):
    uploaded_files = st.file_uploader(
        "Upload any documents (.pdf, .docx, .txt, .md, .csv, .json, code files):",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md", "csv", "json", "py", "js", "html"]
    )

    if uploaded_files:
        if st.button("🚀 Ingest Uploaded Documents", type="primary"):
            progress_bar = st.progress(0)
            total_added = 0

            for idx, uploaded_file in enumerate(uploaded_files):
                # Save temporarily to disk for parsing
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    num_chunks = engine.ingest_file(tmp_path)
                    total_added += num_chunks
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                progress_bar.progress((idx + 1) / len(uploaded_files))

            st.success(f"Successfully processed {len(uploaded_files)} document(s) into {total_added} vector chunks!")
            st.rerun()

# -------------------------------------------------------------
# CHAT INTERFACE
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for i, src in enumerate(msg["sources"]):
                    score_str = f"Score: {src.get('score', 'N/A')}"
                    source_name = src.get("metadata", {}).get("source", f"Source {i+1}")
                    st.markdown(f"**[{i+1}] {source_name}** (`{score_str}`)")
                    st.text(src["content"])

# User Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        retrieved_sources = []

        # Run stream generator
        for event in engine.generate_response_stream(prompt, top_k=top_k, model_override=selected_model):
            if event["type"] == "sources":
                retrieved_sources = event["sources"]
            elif event["type"] == "token":
                full_response += event["token"]
                message_placeholder.markdown(full_response + "▌")
            elif event["type"] == "done":
                message_placeholder.markdown(full_response)

        # Show sources under the generated answer
        if retrieved_sources:
            with st.expander("🔍 View Retrieved Sources"):
                for i, src in enumerate(retrieved_sources):
                    score_str = f"Score: {src.get('score', 'N/A')}"
                    source_name = src.get("metadata", {}).get("source", f"Source {i+1}")
                    st.markdown(f"**[{i+1}] {source_name}** (`{score_str}`)")
                    st.text(src["content"])

    # Persist assistant response in session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": retrieved_sources
    })

import os
from pathlib import Path
import streamlit as st
from config.settings import settings
from config.constants import ALLOWED_TYPES
from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.research_agent import ResearchAgent
from agents.verification_agent import VerificationAgent
from agents.relevance_checker import RelevanceChecker
from agents.workflow import AgentWorkflow
from utils.logging import logger

st.set_page_config(
    page_title="DocChat - Multi-Agent Document Q&A",
    layout="wide"
)

TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0

with st.sidebar:
    st.title("Configuration")

    st.subheader("WatsonX Settings")
    api_key_input = st.text_input(
        "WatsonX API Key",
        value=settings.WATSONX_APIKEY or os.environ.get("WATSONX_APIKEY", ""),
        type="password",
        help="Optional if credentials exist in environment"
    )
    project_id_input = st.text_input(
        "Project ID",
        value=settings.WATSONX_PROJECT_ID or "skills-network"
    )

    st.subheader("Retrieval Options")
    vector_search_k = st.slider(
        "Retrieved Chunks (k)",
        min_value=1,
        max_value=25,
        value=settings.VECTOR_SEARCH_K,
        step=1
    )
    bm25_weight = st.slider(
        "BM25 Keyword Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(settings.HYBRID_RETRIEVER_WEIGHTS[0]),
        step=0.05
    )
    vector_weight = round(1.0 - bm25_weight, 2)
    st.caption(f"Vector Semantic Weight: {vector_weight}")

    relevance_eval_k = st.slider(
        "Relevance Evaluation Chunks",
        min_value=1,
        max_value=15,
        value=5,
        step=1
    )

    st.subheader("Research Agent Options")
    research_model_id = st.text_input(
        "Research Model ID",
        value="meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    )
    research_temp = st.slider(
        "Research Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05
    )
    research_max_tokens = st.slider(
        "Research Max Tokens",
        min_value=100,
        max_value=1024,
        value=300,
        step=50
    )
    custom_instructions = st.text_area(
        "Custom Prompt Instructions",
        value="",
        placeholder="Optional guidelines for the answer (e.g. Focus on technical specs, keep it concise)"
    )

    st.subheader("Verification Agent Options")
    verification_model_id = st.text_input(
        "Verification Model ID",
        value="ibm/granite-4-h-small"
    )
    verification_temp = st.slider(
        "Verification Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )
    verification_max_tokens = st.slider(
        "Verification Max Tokens",
        min_value=50,
        max_value=512,
        value=200,
        step=25
    )
    max_re_research_cycles = st.slider(
        "Max Re-Research Cycles",
        min_value=0,
        max_value=3,
        value=1,
        step=1
    )

    st.subheader("Cache Management")
    cache_expire_days = st.number_input(
        "Cache Expiration (Days)",
        min_value=1,
        max_value=30,
        value=settings.CACHE_EXPIRE_DAYS
    )
    if st.button("Clear Document Cache"):
        processor = DocumentProcessor(cache_expire_days=int(cache_expire_days))
        deleted_count = processor.clear_cache()
        st.success(f"Cleared {deleted_count} cached document files.")

    if st.button("Reset Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("DocChat: Document Question Answering")
st.markdown("Multi-agent question answering with hybrid retrieval, relevance checking, and automated verification.")

with st.expander("Document Management", expanded=(st.session_state.retriever is None)):
    st.info(
        "First-Time Setup Note: When processing a PDF for the first time, Docling automatically "
        "downloads layout and parsing models (~500 MB) from Hugging Face. This download happens "
        "only once and the files are permanently cached on your machine (~/.cache/huggingface). "
        "All subsequent runs use the local cache without redownloading. Ensure a stable internet "
        "connection during the initial PDF run."
    )

    uploaded_files = st.file_uploader(
        "Upload reference documents",
        type=[ext.lstrip(".") for ext in ALLOWED_TYPES],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, MD"
    )

    if uploaded_files:
        if st.button("Process and Index Documents"):
            with st.spinner("Processing documents with Docling and building index..."):
                try:
                    saved_paths = []
                    for uploaded_file in uploaded_files:
                        target_path = TEMP_UPLOAD_DIR / uploaded_file.name
                        with open(target_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_paths.append(target_path)

                    processor = DocumentProcessor(cache_expire_days=int(cache_expire_days))
                    chunks = processor.process(saved_paths)

                    if not chunks:
                        error_details = ""
                        has_network_issue = False
                        if processor.last_errors:
                            lines = []
                            for fname, err in processor.last_errors:
                                lines.append(f"- {fname}: {err}")
                                if any(term in err.lower() for term in ["timed out", "resolution", "connection", "safetensors", "huggingface"]):
                                    has_network_issue = True
                            error_details = "\n" + "\n".join(lines)

                        base_msg = "No valid text chunks could be extracted from uploaded files."
                        if has_network_issue:
                            st.error(
                                f"{base_msg}\n\n"
                                "Network error detected during initial model download from Hugging Face.\n"
                                f"Details:{error_details}\n\n"
                                "Please check your internet connection or DNS and click 'Process and Index Documents' "
                                "again to resume and complete the one-time model download."
                            )
                        else:
                            st.error(f"{base_msg}{error_details}")
                    else:
                        builder = RetrieverBuilder(
                            project_id=project_id_input,
                            api_key=api_key_input if api_key_input else None
                        )
                        retriever = builder.build_hybrid_retriever(
                            docs=chunks,
                            k=vector_search_k,
                            weights=[bm25_weight, vector_weight]
                        )

                        researcher = ResearchAgent(
                            model_id=research_model_id,
                            temperature=research_temp,
                            max_tokens=research_max_tokens,
                            project_id=project_id_input,
                            api_key=api_key_input if api_key_input else None
                        )
                        verifier = VerificationAgent(
                            model_id=verification_model_id,
                            temperature=verification_temp,
                            max_tokens=verification_max_tokens,
                            project_id=project_id_input,
                            api_key=api_key_input if api_key_input else None
                        )
                        relevance_checker = RelevanceChecker(
                            temperature=0.0,
                            project_id=project_id_input,
                            api_key=api_key_input if api_key_input else None
                        )

                        workflow = AgentWorkflow(
                            researcher=researcher,
                            verifier=verifier,
                            relevance_checker=relevance_checker,
                            max_re_research_cycles=max_re_research_cycles,
                            relevance_k=relevance_eval_k
                        )

                        st.session_state.retriever = retriever
                        st.session_state.workflow = workflow
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        st.session_state.num_chunks = len(chunks)
                        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s). Ready for chat.")

                except Exception as e:
                    logger.error(f"Error during ingestion: {e}")
                    st.error(f"Failed to process documents: {e}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Indexed Documents", len(st.session_state.processed_files))
with col2:
    st.metric("Total Extracted Chunks", st.session_state.num_chunks)
with col3:
    status_text = "Active" if st.session_state.retriever is not None else "Awaiting Documents"
    st.metric("Retriever Status", status_text)

if st.session_state.processed_files:
    st.caption("Active files: " + ", ".join(st.session_state.processed_files))

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "verification_report" in message and message["verification_report"]:
            with st.expander("Verification Report"):
                st.markdown(message["verification_report"])
        if "retrieved_sources" in message and message["retrieved_sources"]:
            with st.expander("Retrieved Sources"):
                for idx, doc in enumerate(message["retrieved_sources"]):
                    source_name = doc.metadata.get("source", "Unknown")
                    st.markdown(f"**Chunk {idx + 1} | Source: {source_name}**")
                    st.text(doc.page_content)
                    st.divider()

user_query = st.chat_input("Ask a question about the uploaded documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    if st.session_state.workflow is None or st.session_state.retriever is None:
        response_text = "Please upload and process documents before asking questions."
        with st.chat_message("assistant"):
            st.warning(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing relevance, researching context, and verifying answer..."):
                try:
                    pipeline_result = st.session_state.workflow.full_pipeline(
                        question=user_query,
                        retriever=st.session_state.retriever,
                        custom_instructions=custom_instructions.strip() if custom_instructions else None
                    )

                    answer = pipeline_result.get("draft_answer", "No answer generated.")
                    report = pipeline_result.get("verification_report", "")
                    sources = pipeline_result.get("retrieved_documents", [])

                    st.write(answer)

                    if report:
                        with st.expander("Verification Report"):
                            st.markdown(report)

                    if sources:
                        with st.expander("Retrieved Sources"):
                            for idx, doc in enumerate(sources):
                                source_name = doc.metadata.get("source", "Unknown")
                                st.markdown(f"**Chunk {idx + 1} | Source: {source_name}**")
                                st.text(doc.page_content)
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "verification_report": report,
                        "retrieved_sources": sources
                    })

                except Exception as e:
                    logger.error(f"Error answering query: {e}")
                    error_msg = f"An error occurred while answering your question: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

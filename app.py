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

    st.subheader("Model Provider")
    provider_choice = st.radio(
        "Choose AI Service Provider",
        options=["WatsonX AI", "OpenAI", "Custom API (OpenAI-Compatible)"],
        index=0,
        help="Select whether to use IBM WatsonX, OpenAI, or a Custom OpenAI-compatible API for embeddings and agent reasoning."
    )

    if provider_choice == "WatsonX AI":
        provider_key = "watsonx"
        st.subheader("WatsonX Settings")
        watson_api_key = st.text_input(
            "WatsonX API Key",
            value=settings.WATSONX_APIKEY or os.environ.get("WATSONX_APIKEY", ""),
            type="password",
            help="IBM Cloud API key required when running outside of IBM Skills Network"
        )
        watson_project_id = st.text_input(
            "Project ID",
            value=settings.WATSONX_PROJECT_ID or "skills-network",
            help="Use your WatsonX Project GUID if using a personal IBM Cloud account"
        )
        watson_url = st.text_input(
            "WatsonX API URL",
            value=settings.WATSONX_URL or "https://us-south.ml.cloud.ibm.com",
            help="WatsonX instance URL"
        )

        watson_model_presets = [
            "ibm/granite-3-8b-instruct",
            "meta-llama/llama-3-8b-instruct",
            "meta-llama/llama-3-3-70b-instruct",
            "ibm/granite-13b-chat-v2",
            "ibm/granite-4-h-small",
            "Custom Model..."
        ]
        selected_preset = st.selectbox(
            "Model Selection",
            options=watson_model_presets,
            index=0,
            help="Select a standard/trial model or choose 'Custom Model...' to enter your own."
        )
        if selected_preset == "Custom Model...":
            custom_model_input = st.text_input("Custom Model ID", value="", placeholder="e.g. ibm/granite-3-8b-instruct")
            active_model_id = custom_model_input.strip() or "ibm/granite-3-8b-instruct"
        else:
            active_model_id = selected_preset

        active_api_key = watson_api_key
        active_project_id = watson_project_id
        active_api_url = watson_url

    elif provider_choice == "OpenAI":
        provider_key = "openai"
        st.subheader("OpenAI Settings")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="OpenAI API key (sk-...)"
        )
        openai_base_url = st.text_input(
            "API Base URL",
            value=settings.OPENAI_BASE_URL or "https://api.openai.com/v1",
            help="OpenAI API Base URL (or custom proxy/gateway)"
        )

        openai_model_presets = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
            "Custom Model..."
        ]
        selected_preset = st.selectbox(
            "Model Selection",
            options=openai_model_presets,
            index=0,
            help="Select a standard/trial model or choose 'Custom Model...' to enter your own."
        )
        if selected_preset == "Custom Model...":
            custom_model_input = st.text_input("Custom Model ID", value="", placeholder="e.g. gpt-4o-mini")
            active_model_id = custom_model_input.strip() or "gpt-4o-mini"
        else:
            active_model_id = selected_preset

        active_api_key = openai_api_key
        active_project_id = None
        active_api_url = openai_base_url

    else:
        provider_key = "custom"
        st.subheader("Custom API Settings")
        custom_api_key = st.text_input(
            "Custom API Key",
            value=settings.CUSTOM_API_KEY or os.environ.get("CUSTOM_API_KEY", ""),
            type="password",
            help="API key for custom provider (leave empty if not required)"
        )
        custom_api_url = st.text_input(
            "Custom API Base URL",
            value=settings.CUSTOM_API_URL or os.environ.get("CUSTOM_API_URL", "https://api.groq.com/openai/v1"),
            help="OpenAI-compatible base URL (e.g. https://api.groq.com/openai/v1, https://openrouter.ai/api/v1, http://localhost:11434/v1)"
        )

        custom_model_presets = [
            "llama-3.3-70b-versatile",
            "meta-llama/llama-3.3-70b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "deepseek/deepseek-r1:free",
            "Custom Model..."
        ]
        selected_preset = st.selectbox(
            "Model Selection",
            options=custom_model_presets,
            index=0,
            help="Select a free trial model or choose 'Custom Model...' to enter your own."
        )
        if selected_preset == "Custom Model...":
            custom_model_input = st.text_input("Custom Model ID", value="", placeholder="e.g. meta-llama/llama-3.3-70b-instruct:free")
            active_model_id = custom_model_input.strip() or "llama-3.3-70b-versatile"
        else:
            active_model_id = selected_preset

        active_api_key = custom_api_key
        active_project_id = None
        active_api_url = custom_api_url

    research_model_default = active_model_id
    verification_model_default = active_model_id

    st.subheader("Embedding Model (Vector Search)")
    embedding_source = st.radio(
        "Embedding Source",
        options=["Local Offline Model (Free, Cached)", "Cloud API Provider"],
        index=0,
        help="Local models download once and run 100% offline on your machine with 0 API keys and zero cost. Cloud models use the remote API."
    )

    if embedding_source == "Local Offline Model (Free, Cached)":
        embedding_type = "local"
        local_model_presets = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            "Custom Local HuggingFace Model..."
        ]
        selected_local_emb = st.selectbox(
            "Select Local Embedding Model",
            options=local_model_presets,
            index=0,
            help="Downloaded once to ~/.cache/huggingface and cached permanently. Runs fast on CPU."
        )
        if selected_local_emb == "Custom Local HuggingFace Model...":
            custom_local_emb = st.text_input("Enter HuggingFace Model ID", value="", placeholder="e.g. BAAI/bge-small-en-v1.5")
            active_embedding_model = custom_local_emb.strip() or "sentence-transformers/all-MiniLM-L6-v2"
        else:
            active_embedding_model = selected_local_emb
    else:
        embedding_type = "cloud"
        if provider_key == "watsonx":
            cloud_default_emb = "ibm/granite-embedding-278m-multilingual"
        else:
            cloud_default_emb = "text-embedding-3-small"
        active_embedding_model = st.text_input(
            "Cloud Embedding Model ID",
            value=cloud_default_emb,
            help="Model name for remote embedding API (requires an API that supports /v1/embeddings)"
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
        value=research_model_default
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
        value=verification_model_default
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
        "Setup Note: When processing a PDF for the first time, Docling downloads required "
        "layout models (~500 MB) from Hugging Face once and permanently caches them in ~/.cache/huggingface. "
        "Text (.txt) and Markdown (.md) files are read directly without downloading models."
    )

    uploaded_files = st.file_uploader(
        "Upload reference documents",
        type=[ext.lstrip(".") for ext in ALLOWED_TYPES],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, MD"
    )

    if uploaded_files:
        if st.button("Process and Index Documents"):
            with st.status("Processing and indexing documents...", expanded=True) as status:
                try:
                    clean_api_key = active_api_key.strip() if active_api_key else ""
                    clean_api_url = active_api_url.strip() if active_api_url else None
                    if not clean_api_key and provider_key != "custom":
                        status.update(label="Authentication error: Missing API Key", state="error")
                        st.error(
                            f"Please provide an API key for {provider_choice} in the sidebar configuration "
                            f"(or set it in your .env file) before processing documents."
                        )
                        st.stop()
                    if not clean_api_key and provider_key == "custom":
                        clean_api_key = "dummy-key"

                    status.write("Step 1: Saving and parsing uploaded files with Docling...")
                    saved_paths = []
                    for uploaded_file in uploaded_files:
                        target_path = TEMP_UPLOAD_DIR / uploaded_file.name
                        with open(target_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_paths.append(target_path)

                    processor = DocumentProcessor(cache_expire_days=int(cache_expire_days))
                    chunks = processor.process(saved_paths)

                    if not chunks:
                        status.update(label="Processing failed: No chunks extracted", state="error")
                        error_details = ""
                        has_network_issue = False
                        if processor.last_errors:
                            lines = []
                            for fname, err in processor.last_errors:
                                lines.append(f"- {fname}: {err}")
                                if any(term in err.lower() for term in ["timed out", "resolution", "connection", "safetensors", "huggingface"]):
                                    has_network_issue = True
                            error_details = "\n" + "\n".join(lines)

                        base_msg = "No text chunks could be extracted from the uploaded files."
                        if has_network_issue:
                            st.error(
                                f"{base_msg}\n\n"
                                "Network connection issue detected while downloading Docling models.\n"
                                f"Details:{error_details}\n\n"
                                "Please ensure your internet connection is active and click 'Process and Index Documents' again."
                            )
                        else:
                            st.error(f"{base_msg}{error_details}")
                        st.stop()

                    status.write(f"Step 1 Complete: Extracted {len(chunks)} text chunks across {len(saved_paths)} file(s).")

                    if embedding_type == "local":
                        status.write(f"Step 2: Loading local embedding model '{active_embedding_model}' (cached offline)...")
                    else:
                        status.write(f"Step 2: Connecting to {provider_choice} cloud embeddings ({active_embedding_model})...")

                    builder = RetrieverBuilder(
                        provider=provider_key,
                        model_id=active_embedding_model.strip() if active_embedding_model else None,
                        url=clean_api_url,
                        project_id=active_project_id,
                        api_key=clean_api_key,
                        embedding_source=embedding_type
                    )

                    status.write("Step 3: Creating BM25 keyword index and building hybrid ensemble...")
                    retriever = builder.build_hybrid_retriever(
                        docs=chunks,
                        k=vector_search_k,
                        weights=[bm25_weight, vector_weight]
                    )

                    status.write(f"Step 4: Connecting to {provider_choice} models and initializing multi-agent workflow...")
                    effective_research_model = research_model_id.strip() if research_model_id else active_model_id
                    effective_verification_model = verification_model_id.strip() if verification_model_id else active_model_id
                    effective_relevance_model = effective_research_model

                    researcher = ResearchAgent(
                        provider=provider_key,
                        model_id=effective_research_model,
                        temperature=research_temp,
                        max_tokens=research_max_tokens,
                        project_id=active_project_id,
                        api_key=clean_api_key,
                        url=clean_api_url
                    )
                    verifier = VerificationAgent(
                        provider=provider_key,
                        model_id=effective_verification_model,
                        temperature=verification_temp,
                        max_tokens=verification_max_tokens,
                        project_id=active_project_id,
                        api_key=clean_api_key,
                        url=clean_api_url
                    )
                    relevance_checker = RelevanceChecker(
                        provider=provider_key,
                        model_id=effective_relevance_model,
                        temperature=0.0,
                        project_id=active_project_id,
                        api_key=clean_api_key,
                        url=clean_api_url
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

                    status.update(
                        label=f"Indexing complete: {len(chunks)} chunks ready from {len(uploaded_files)} file(s).",
                        state="complete",
                        expanded=False
                    )
                    st.success(f"System ready using {provider_choice}. You can now ask questions below.")

                except Exception as e:
                    err_str = str(e)
                    logger.error(f"Error during ingestion: {err_str}")
                    status.update(label="Error during indexing", state="error")
                    if "IAM Token" in err_str or "Response [400]" in err_str:
                        st.error(
                            "Failed to authenticate with IBM Cloud (IAM Token Response 400):\n\n"
                            "1. Your WatsonX API Key is missing or invalid. Please verify the key in the sidebar.\n"
                            "2. If using your own IBM Cloud account, change 'skills-network' to your WatsonX Project ID.\n"
                            "3. Alternatively, switch the 'Model Provider' to OpenAI in the sidebar if you have an OpenAI key."
                        )
                    else:
                        st.error(f"Failed to process documents: {e}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Indexed Documents", len(st.session_state.processed_files))
with col2:
    st.metric("Total Extracted Chunks", st.session_state.num_chunks)
with col3:
    emb_display = active_embedding_model.split('/')[-1]
    status_text = f"Active ({active_model_id} | {emb_display})" if st.session_state.retriever is not None else "Awaiting Documents"
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
            with st.status("Analyzing and answering question...", expanded=True) as query_status:
                try:
                    clean_api_key = active_api_key.strip() if active_api_key else ("dummy-key" if provider_key == "custom" else "")
                    clean_api_url = active_api_url.strip() if active_api_url else None
                    effective_research_model = research_model_id.strip() if research_model_id else active_model_id
                    effective_verification_model = verification_model_id.strip() if verification_model_id else active_model_id
                    effective_relevance_model = effective_research_model

                    # Re-instantiate agents with current UI settings so sidebar changes take effect immediately
                    researcher = ResearchAgent(
                        provider=provider_key,
                        model_id=effective_research_model,
                        temperature=research_temp,
                        max_tokens=research_max_tokens,
                        project_id=active_project_id,
                        api_key=clean_api_key,
                        url=clean_api_url
                    )
                    verifier = VerificationAgent(
                        provider=provider_key,
                        model_id=effective_verification_model,
                        temperature=verification_temp,
                        max_tokens=verification_max_tokens,
                        project_id=active_project_id,
                        api_key=clean_api_key,
                        url=clean_api_url
                    )
                    relevance_checker = RelevanceChecker(
                        provider=provider_key,
                        model_id=effective_relevance_model,
                        temperature=0.0,
                        project_id=active_project_id,
                        api_key=clean_api_key,
                        url=clean_api_url
                    )

                    workflow = AgentWorkflow(
                        researcher=researcher,
                        verifier=verifier,
                        relevance_checker=relevance_checker,
                        max_re_research_cycles=max_re_research_cycles,
                        relevance_k=relevance_eval_k
                    )
                    st.session_state.workflow = workflow

                    pipeline_result = st.session_state.workflow.full_pipeline(
                        question=user_query,
                        retriever=st.session_state.retriever,
                        custom_instructions=custom_instructions.strip() if custom_instructions else None,
                        step_callback=lambda msg: query_status.write(msg)
                    )

                    answer = pipeline_result.get("draft_answer", "No answer generated.")
                    report = pipeline_result.get("verification_report", "")
                    sources = pipeline_result.get("retrieved_documents", [])

                    query_status.update(label="Complete", state="complete", expanded=False)

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
                    query_status.update(label="Query execution error", state="error")
                    error_msg = f"An error occurred while answering your question: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

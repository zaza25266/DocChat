# DocChat: Multi-Agent Document Question-Answering System

DocChat is an end-to-end Retrieval-Augmented Generation (RAG) platform designed to answer questions strictly grounded in user-provided documents. Rather than relying on a single large language model (LLM) query that is prone to hallucination, DocChat implements a modular, multi-agent architecture orchestrated via LangGraph, backed by a hybrid retrieval engine (BM25 lexical search combined with Chroma dense vector search) and a document conversion pipeline powered by IBM Docling.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture and Design](#architecture-and-design)
3. [End-to-End Data Flow](#end-to-end-data-flow)
4. [Component Deep Dive](#component-deep-dive)
   - [Document Processing and Caching](#1-document-processing-and-caching)
   - [Hybrid Retrieval Engine](#2-hybrid-retrieval-engine)
   - [Multi-Agent Orchestration](#3-multi-agent-orchestration)
5. [Configurable Parameters and User Inputs](#configurable-parameters-and-user-inputs)
6. [Core RAG and Multi-Agent Concepts for Students](#core-rag-and-multi-agent-concepts-for-students)
7. [Installation and Setup](#installation-and-setup)
8. [Running the Application](#running-the-application)
9. [Project Directory Structure](#project-directory-structure)

---

## System Overview

Standard LLM applications often suffer from hallucinations—fabricating facts when asked questions outside their training data or when tasked with analyzing complex private files. DocChat addresses this with three engineering principles:

- **Grounded Context**: The model only generates responses from passages extracted directly from your files.
- **Relevance Gatekeeping**: Queries are screened before generation to confirm that the uploaded documents actually contain relevant data.
- **Automated Verification and Self-Correction**: A separate verification agent inspects the drafted answer against the source context to detect unsupported claims or contradictions. If issues are identified, the workflow triggers a re-research cycle.

---

## Architecture and Design

The system consists of three distinct subsystems:

1. **Ingestion Subsystem**: Ingests heterogeneous files (PDF, DOCX, TXT, Markdown), converts them into clean structured Markdown using IBM Docling, partitions text into hierarchical sections, and stores content hashes for rapid disk-cached retrieval.
2. **Retrieval Subsystem**: Creates an ensemble search index combining lexical keyword matching (BM25) and dense semantic vector similarity (ChromaDB + local offline cached embeddings or cloud provider embeddings).
3. **Multi-Agent LangGraph Subsystem**: Executes a state machine consisting of Relevance Checking, Research Generation, and Verification Fact-Checking.

```
                      +-----------------------------+
                      |   User Document Uploads     |
                      |  (.pdf, .docx, .txt, .md)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     DocumentProcessor       |
                      |  - SHA-256 Cache Check      |
                      |  - Docling Markdown Parse   |
                      |  - Header-Based Chunking    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |      RetrieverBuilder       |
                      |  - BM25 Lexical Index       |
                      |  - Chroma Vector Store      |
                      |  - EnsembleRetriever Fusion |
                      +--------------+--------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
    +--------------------+                       +--------------------+
    |    User Query      |                       | Hybrid Retriever   |
    +----------+---------+                       +----------+---------+
               |                                            |
               +---------------------+----------------------+
                                     |
                                     v
                       +---------------------------+
                       |   AgentWorkflow (State)   |
                       +-------------+-------------+
                                     |
                                     v
                       +---------------------------+
                       |    Relevance Checker      |
                       +-------------+-------------+
                                     |
                      +--------------+--------------+
         [NO_MATCH]   |                             | [CAN_ANSWER / PARTIAL]
                      v                             v
           +--------------------+      +---------------------------+
           | Early Exit Message |      |      Research Agent       |
           +--------------------+      +-------------+-------------+
                                                     |
                                                     v
                                       +---------------------------+
                                       |    Verification Agent     |
                                       +-------------+-------------+
                                                     |
                                   +-----------------+-----------------+
                    [Supported: NO |                   [Supported: YES |
                     Iterations <= max]                 Iterations > max]
                                   v                                   v
                     +---------------------------+       +---------------------------+
                     | Re-Research Feedback Loop |       | Return Verified Answer    |
                     +---------------------------+       | + Audit Report to User    |
                                                         +---------------------------+
```

---

## End-to-End Data Flow

The lifecycle of a question in DocChat proceeds through five distinct stages:

1. **File Upload**: The user uploads one or more reference files via the Streamlit interface. Files are saved locally to `temp_uploads/`.
2. **Parsing and Chunking**: `DocumentProcessor` calculates the SHA-256 hash of each file. If a cache entry exists in `document_cache/`, chunks are loaded immediately. Otherwise, Docling parses the document structure into Markdown, and `MarkdownHeaderTextSplitter` partitions the text based on `#` and `##` headings.
3. **Indexing**: `RetrieverBuilder` generates embeddings using either a local offline cached model (e.g. `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, downloaded once and cached permanently) or a cloud embedding API (WatsonX, OpenAI), loads the chunk representations into ChromaDB, constructs a BM25 index, and joins them using LangChain's `EnsembleRetriever`.
4. **Agent Workflow Execution**:
   - The user inputs a query.
   - `EnsembleRetriever` fetches the top candidate chunks.
   - **Relevance Checker**: Evaluates whether the retrieved context contains relevant information using the configured model. If classified as `NO_MATCH`, processing stops, informing the user that the query cannot be answered from the document.
   - **Research Agent**: If relevant, the research agent compiles the passages and user instructions into a focused prompt and invokes the generation LLM (e.g. your selected model or custom free trial model).
   - **Verification Agent**: Receives the drafted response and the source passages, checking for direct support, unsupported claims, contradictions, and overall relevance.
   - If the verification report indicates unsupported claims or contradictions, and the retry limit has not been reached, the state transitions back to the research node.
5. **UI Rendering**: Streamlit renders the generated answer, an expandable Verification Report detailing the factual consistency audit, and an expandable Retrieved Sources view showing the exact document passages used.

---

## Component Deep Dive

### 1. Document Processing and Caching
Located in `document_processor/file_handler.py`.

- **Fast PDF and Document Processing**: Digital PDFs are parsed directly with sub-second text extraction and split along page and section headers. Scanned image PDFs automatically fall back to IBM Docling's OCR and layout models.
- **Direct Text/Markdown Ingestion**: For `.txt` and `.md` files, DocChat reads the text directly and applies header splitting, bypassing heavy neural models for maximum speed.
- **Markdown Header Splitting**: Rather than cutting text arbitrarily at a fixed character count (which can split sentences or tables in half), `MarkdownHeaderTextSplitter` splits text along logical heading boundaries (`#` and `##`), ensuring each chunk maintains semantic coherence.
- **Content Hash Caching**: Repeatedly parsing large documents is computationally expensive. DocChat computes a SHA-256 hash of the binary file content:
  - If a `.pkl` cache file matching the hash exists and has not expired, chunks are loaded from disk in milliseconds.
  - If the file is modified or new, it is processed and cached for future queries.

### 2. Hybrid Retrieval Engine
Located in `retriever/builder.py`.

Hybrid search combines the strengths of two complementary information retrieval methods:

- **Lexical Search (BM25)**:
  - Works through exact token matching, term frequency, and inverse document frequency.
  - Highly effective for technical keywords, serial numbers, product codes, abbreviations, and rare proper nouns that dense embedding models might dilute.
- **Dense Vector Search (ChromaDB + Local or Cloud Embeddings)**:
  - **Local Offline Embeddings (Default)**: Downloads a lightweight model (~80 MB, e.g. `sentence-transformers/all-MiniLM-L6-v2`) on first use and permanently caches it in `~/.cache/huggingface`. Runs 100% locally on CPU with zero API costs, zero quota limits, and no 501/429 errors from chat-only LLM endpoints (like Groq, Gemini, or OpenRouter).
  - **Cloud Embeddings**: Can alternatively use IBM WatsonX Granite embeddings or OpenAI `text-embedding-3-small` if cloud embeddings are desired.
  - Converts text chunks into mathematical vectors in a continuous geometric space for conceptual similarity.
- **Ensemble Fusion**:
  - LangChain's `EnsembleRetriever` normalizes the rank scores of both retrievers using reciprocal rank weighting.
  - Users can adjust the relative weights in the UI (e.g., `0.4` BM25 weight and `0.6` Vector weight).

### 3. Multi-Agent Orchestration
Located in `agents/`.

DocChat divides reasoning into specialized agents coordinated by a LangGraph state graph (`AgentWorkflow`):

- **Relevance Checker (`relevance_checker.py`)**:
  - Inspects the top retrieved passages and classifies question relevance into `CAN_ANSWER`, `PARTIAL`, or `NO_MATCH`.
  - Acts as an early guardrail. If no relevant data is present, the workflow terminates early, saving LLM tokens and avoiding hallucinations.
- **Research Agent (`research_agent.py`)**:
  - Focuses solely on synthesizing an accurate, comprehensive answer from the context passages.
  - Uses temperature controls and system instructions to minimize creativity and maximize grounding.
- **Verification Agent (`verification_agent.py`)**:
  - Operates as an independent auditor.
  - Analyzes the drafted answer against the context and outputs a structured assessment:
    - `Supported: YES/NO`
    - `Unsupported Claims: [list]`
    - `Contradictions: [list]`
    - `Relevant: YES/NO`
    - `Additional Details: text`
- **Feedback Loop**:
  - If the answer fails verification, the workflow routes back to the research agent with the verification context, enforcing an automated self-correction mechanism up to `max_re_research_cycles`.

---

## Configurable Parameters and User Inputs

DocChat exposes configurable options directly within the Streamlit sidebar, allowing students and practitioners to observe the impact of different hyperparameters:

| Parameter | Default | Description | Impact of Modification |
|---|---|---|---|
| **AI Service Provider** | `WatsonX AI` | Choose between WatsonX AI, OpenAI, or Custom API (OpenAI-Compatible) | Selects the active LLM brain for reasoning agents (Research, Verification, Relevance). |
| **API Base URL / Endpoint** | Provider default | Custom API endpoint or proxy URL | Allows pointing to alternative WatsonX regions, OpenAI proxies, or OpenAI-compatible trial services (e.g., Groq, OpenRouter, Ollama). |
| **Model Selection** | First preset | Preset dropdown with a "Custom Model..." write-in option | Choose from popular models or enter any custom / free trial model ID available on your account. |
| **Embedding Source** | `Local Offline Model (Free, Cached)` | Local offline models vs. Cloud API provider | Local models run offline with zero API keys and zero cost. Cloud uses the remote provider API. |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Model used for semantic vector indexing | Local model (~80 MB) downloads on first use and caches permanently in `~/.cache/huggingface`. |
| **Retrieved Chunks (k)** | `10` | Number of candidate chunks returned by the retriever | Higher values provide broader context but may introduce irrelevant noise or exceed token limits. |
| **BM25 Keyword Weight** | `0.4` | Weight given to lexical keyword retrieval | Increase when working with code, IDs, technical specs, or strict keyword queries. |
| **Vector Semantic Weight** | `0.6` | Weight given to dense semantic embedding retrieval | Increase when queries involve conceptual reasoning, paraphrasing, or abstract questions. |
| **Relevance Evaluation Chunks** | `5` | Number of top passages evaluated by the Relevance Checker | Controls how much context the relevance classifier reads before making a decision. |
| **Research Model ID** | Selected Model | Model ID used by the Research Agent for answer drafting | Uses the selected custom/preset model by default; can be individually customized. |
| **Research Temperature** | `0.3` | Randomness in research generation | Lower values (0.0 - 0.3) yield deterministic, factual output. Higher values increase variability. |
| **Research Max Tokens** | `300` | Maximum token length for the draft answer | Controls the maximum length of generated answers. |
| **Verification Model ID** | Selected Model | Model ID used by the Verification Agent for consistency auditing | Uses the selected custom/preset model by default; can be individually customized. |
| **Verification Temperature** | `0.0` | Randomness in verification auditing | Kept at 0.0 for consistent, reproducible evaluation. |
| **Max Re-Research Cycles** | `1` | Maximum retry attempts if verification fails | Prevents infinite loops while allowing self-correction. |
| **Enable Verification Audit** | `True` | Toggle multi-agent factual consistency check | Disable for Fast Mode (single-pass generation, ~3x faster responses). |
| **Enable Relevance Gatekeeper** | `True` | Toggle relevance screening before synthesis | Prevents hallucinating on unrelated topics; auto-passes overview/summary queries. |
| **Max Context Chunks to LLM** | `5` | Top chunks sent into the LLM context prompt | Controls prompt token volume; lower values (3-5) dramatically accelerate answer generation. |
| **Custom Prompt Instructions** | `""` | User-defined constraints for generation | Inject guidelines such as "Focus on numeric metrics" or "Format as bullet points". |
| **Cache Expiration (Days)** | `7` | Time before cached document pickles expire | Controls how long processed chunks are preserved on disk. |

---

## Core RAG and Multi-Agent Concepts for Students

### Why RAG Over Fine-Tuning?
- **Knowledge Updates**: Updating a RAG system requires adding a document to the index; fine-tuning requires expensive re-training.
- **Traceability**: RAG systems can cite the exact chunks used to formulate the answer.
- **Access Control and Privacy**: Documents can be segmented, deleted, or scoped without modifying model weights.

### Dense vs. Sparse Retrieval
- **Sparse (BM25)** relies on exact token occurrence. It fails when a user asks about "automobiles" while the document only mentions "cars".
- **Dense Embeddings** map semantics to vectors. They understand that "automobile" and "car" are related, but can struggle with precise alphanumeric strings (e.g., "Error code 404-XJ").
- **Hybrid Search** combines both methods, ensuring queries with both conceptual meaning and technical keywords return the most accurate passages.

### Multi-Agent Decomposition
In traditional RAG pipelines, an LLM performs generation without validation. In DocChat, separation of concerns prevents confirmation bias:
- Generating an answer requires synthesis and fluency.
- Verifying an answer requires critical fact checking and contradiction detection.
- Decoupling these tasks into separate agent nodes with specialized prompts and temperatures significantly reduces error rates.

---

## Installation and Setup

### Prerequisites
- Python 3.10 or Python 3.11
- Operating System: macOS, Linux, or Windows (via WSL or PowerShell)

### 1. Clone the Repository
```bash
git clone <repository_url>
cd DocChat
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` in an editor and set your credentials:
```env
# Optional WatsonX Settings (can also be entered in the UI)
WATSONX_APIKEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=skills-network
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Optional OpenAI Settings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional Custom API Settings (e.g. Groq, OpenRouter, Together, Ollama)
CUSTOM_API_KEY=your_custom_api_key_here
CUSTOM_API_URL=https://api.groq.com/openai/v1

# Storage and Retrieval defaults
CHROMA_DB_PATH=./chroma_db
CHROMA_COLLECTION_NAME=documents
VECTOR_SEARCH_K=10
LOG_LEVEL=INFO
```

---

## Recommended Free Trial Providers & Models

If you do not have active OpenAI or WatsonX credits, use any of these free trial providers:

| Provider | Base URL | Recommended Models | Notes |
|---|---|---|---|
| **Groq (Recommended)** | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile`<br>`llama-3.1-8b-instant` | **Fastest option** (~500+ tok/s). Free API keys at [console.groq.com](https://console.groq.com). |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-1.5-flash`<br>`gemini-2.0-flash-exp` | Stable official models. Free API keys at [aistudio.google.com](https://aistudio.google.com). |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.3-70b-instruct:free`<br>`mistralai/mistral-7b-instruct:free` | Free tier models available at [openrouter.ai](https://openrouter.ai). |
| **Ollama (100% Local)** | `http://localhost:11434/v1` | `llama3.2`, `mistral`, `qwen2.5` | Completely offline, private, zero cost, no API key needed. |

> [!TIP]
> **Why was the app slow previously?**
> A full multi-agent RAG pipeline runs 3 sequential LLM calls per query: (1) Relevance Gatekeeper, (2) Research Agent, and (3) Verification Agent. If sending 10-15 full chunks to slow remote endpoints, this can take 15-30 seconds.
> **How to make it blazing fast:**
> 1. Use **Groq** (`llama-3.3-70b-versatile` or `llama-3.1-8b-instant`).
> 2. Keep **Local Offline Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) selected.
> 3. Set **Max Context Chunks to LLM** to `4` or `5`.
> 4. For instant answers, uncheck **Enable Verification Audit** (switches to single-pass Fast Mode).

---

## Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will start and open automatically in your browser (default URL: `http://localhost:8501`).

### How to Use DocChat:
1. **Configure Parameters**: Expand the configuration sections in the sidebar to review or adjust retrieval weights, temperatures, and model IDs.
2. **Upload Documents**: Under the "Document Management" expander in the main area, upload your files (PDF, DOCX, TXT, or Markdown).
3. **Index Documents**: Click **Process and Index Documents**. The system will convert documents, extract chunks, build the hybrid retriever, and initialize the agent workflow.
4. **Ask Questions**: Type your question into the chat input bar at the bottom.
5. **Inspect Results**:
   - Read the assistant's answer.
   - Click **Verification Report** to review the factual consistency assessment.
   - Click **Retrieved Sources** to inspect the source chunks and file names used to generate the answer.

---

## Project Directory Structure

```
DocChat/
├── agents/                         # Multi-agent implementations and workflow
│   ├── __init__.py                 # Exports ResearchAgent, VerificationAgent, etc.
│   ├── relevance_checker.py        # Relevance classification agent
│   ├── research_agent.py           # Answer synthesis agent
│   ├── verification_agent.py       # Fact verification and audit agent
│   └── workflow.py                 # LangGraph state machine orchestrator
├── config/                         # Configuration and constants
│   ├── __init__.py                 # Settings exports
│   ├── constants.py                # File size limits and allowed extensions
│   └── settings.py                 # Pydantic BaseSettings loading from .env
├── document_processor/             # Ingestion and chunking pipeline
│   ├── __init__.py                 # Processor exports
│   └── file_handler.py             # Docling parser, header splitter, cache logic
├── retriever/                      # Retrieval index builders
│   ├── __init__.py                 # RetrieverBuilder and LocalEmbeddings exports
│   ├── builder.py                  # BM25 + Chroma hybrid ensemble builder
│   └── local_embeddings.py         # PyTorch/Transformers offline cached embeddings
├── utils/                          # Utilities and logging
│   ├── __init__.py
│   └── logging.py                  # Loguru logger setup
├── .env.example                    # Environment variable template
├── app.py                          # Streamlit interactive application
├── requirements.txt                # Python package dependencies
└── README.md                       # System documentation and learning guide
```

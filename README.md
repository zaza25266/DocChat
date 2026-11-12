# DocChat: Multi-Agent Document Q&A

DocChat is a fast, multi-agent Retrieval-Augmented Generation (RAG) platform. It provides grounded, factual answers from your files (PDF, DOCX, TXT, MD) using hybrid search and automated fact-verification.

---

## Quickstart

```bash
# 1. Clone & enter repository
git clone <repository_url> && cd DocChat

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate
# Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip && pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Architecture

```
User Query
    │
    ▼
[Hybrid Retrieval] ──► BM25 (Keywords) + ChromaDB (Local Dense Vectors)
    │
    ▼
[Agent Workflow]
    ├─► 1. Relevance Checker ─► Fast-passes document topics; screens alien queries
    ├─► 2. Research Agent    ─► Strictly synthesizes answer from context
    └─► 3. Verifier Agent    ─► Audits answer against sources (Self-Correction loop)
    │
    ▼
Verified Answer + Source Passages + Audit Report
```

### Agent Roles
- **Relevance Checker**: Validates query relevance; fast-tracks summaries and direct keywords.
- **Research Agent**: Drafts grounded answers with strict anti-hallucination rules.
- **Verification Agent**: Audits claims against source passages and triggers re-research if unsupported.

---

## Key Features

- **Fast Multi-Format Ingestion**:
  - **PDF**: Sub-second digital text extraction via `pypdf`; auto-falls back to Docling OCR for scanned pages.
  - **DOCX**: Native high-speed extraction via `python-docx` with table and heading preservation.
  - **TXT / MD**: Direct instant ingestion.
- **Chunk Normalization**: Recursive character chunking (`~800` chars, `120` overlap) prevents vector truncation.
- **Free Local Offline Embeddings**: Runs `sentence-transformers/all-MiniLM-L6-v2` (~80 MB) locally on CPU with zero API costs, zero rate limits, and 100% offline privacy.
- **Universal LLM Compatibility**: Plug in any provider (Groq, OpenAI, Google Gemini, OpenRouter, WatsonX, or local Ollama).
- **Flexible Response Depth**: Toggle between **Concise (Short)**, **Balanced (Standard)**, and **In-Depth (Detailed / Long)** directly in the UI.
- **Fast Mode**: Option to bypass verification for instant 1-step responses (~1–2 seconds).

---

## Free Trial Providers & Models

DocChat runs completely free using local embeddings paired with any free-tier LLM API:

| Provider | Base URL | Recommended Model | Highlights |
|---|---|---|---|
| **Groq (Recommended)** | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile`<br>`llama-3.1-8b-instant` | **Blazing fast** (~500–800 tok/s). Free keys at [console.groq.com](https://console.groq.com). |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-1.5-flash`<br>`gemini-2.0-flash-exp` | Official OpenAI endpoint. Free keys at [aistudio.google.com](https://aistudio.google.com). |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.3-70b-instruct:free`<br>`mistralai/mistral-7b-instruct:free` | Free community models at [openrouter.ai](https://openrouter.ai). |
| **Ollama (100% Local)** | `http://localhost:11434/v1` | `llama3.2`, `mistral`, `qwen2.5` | Fully offline, zero cost, no API keys needed. |
| **OpenAI / WatsonX** | Standard endpoints | `gpt-4o-mini`, `granite-3-8b` | Commercial cloud endpoints. |

---

## Configuration & UI Controls

### Main Screen Controls
- **Response Length**: Choose between **Concise** (~400 tokens), **Balanced** (~1024 tokens), or **In-Depth** (~2500 tokens).
- **Fast Mode**: Checkbox to toggle instant single-pass generation vs. full verification audit.

### Sidebar Parameters
| Parameter | Default | Purpose |
|---|---|---|
| **AI Provider** | `WatsonX` / `Custom API` | Selects LLM backend. |
| **Embedding Source** | `Local Offline` | Local CPU MiniLM vs. Remote Cloud API. |
| **Retrieved Chunks ($k$)** | `10` | Total candidate chunks fetched by hybrid search. |
| **BM25 / Vector Weight** | `0.4 / 0.6` | Balance between exact keywords vs. semantic meaning. |
| **Max Context Chunks** | `5` | Top chunks sent into the prompt (lower = faster). |
| **Relevance Gatekeeper** | `Off` | Optional pre-filter for out-of-scope queries. |
| **Max Re-Research** | `1` | Self-correction attempts if verification flags issues. |

---

## Environment Variables (`.env`)

Create a `.env` file from `.env.example` if you prefer saving keys locally:

```env
# Custom / Free Trial API (e.g. Groq, OpenRouter, Ollama)
CUSTOM_API_KEY=your_groq_or_openrouter_key
CUSTOM_API_URL=https://api.groq.com/openai/v1

# OpenAI (Optional)
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1

# IBM WatsonX (Optional)
WATSONX_APIKEY=your_watsonx_key
WATSONX_PROJECT_ID=skills-network
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

---

## Project Structure

```
DocChat/
├── agents/                         # Multi-agent logic & LangGraph workflow
│   ├── relevance_checker.py        # Query screening & keyword matching
│   ├── research_agent.py           # Grounded answer synthesis
│   ├── verification_agent.py       # Factual consistency auditor
│   └── workflow.py                 # Graph orchestrator & retry loops
├── document_processor/             # File parsing & chunking
│   ├── file_handler.py             # PDF (pypdf), DOCX (python-docx), Docling
│   └── __init__.py
├── retriever/                      # Retrieval & indexing
│   ├── builder.py                  # Hybrid BM25 + Chroma ensemble
│   ├── local_embeddings.py         # PyTorch/Transformers offline MiniLM
│   └── __init__.py
├── config/                         # App settings and constants
├── app.py                          # Streamlit user interface
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

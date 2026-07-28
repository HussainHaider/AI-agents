# AI-agents

A hands-on learning repository for building AI agents with **LangChain** and **LangGraph**. It works through the core building blocks of LLM applications — from chat models and prompts to structured output, chains, runnables, retrievers, vector stores, and a full Retrieval-Augmented Generation (RAG) pipeline.

## Repository layout

```
AI-agents/
├── langChain-practice/     # LangChain fundamentals (most of the code lives here)
└── langGraph-practice/     # LangGraph experiments (in progress)
```

### `langChain-practice/`

| Folder | What it covers |
| --- | --- |
| `models/` | Chat models (OpenAI, Anthropic, Google, Hugging Face), LLMs, and embedding models |
| `prompts/` | Prompt templates, chat prompt templates, message placeholders, and a Streamlit prompt UI |
| `structured-output/` | Structured output with `TypedDict`, Pydantic, and JSON schema |
| `output-parsers/` | String, JSON, structured, and Pydantic output parsers |
| `chains/` | Simple, sequential, parallel, and conditional chains |
| `runnables/` | Runnable primitives (sequence, parallel, passthrough, branch, lambda) and LCEL notebooks |
| `document-loaders/` | Text, PDF, CSV, directory, and web-based document loaders |
| `text-splitters/` | Length-, structure-, and semantic-based splitting for text, Markdown, and code |
| `vector-stores/` | Chroma vector store usage |
| `retrievers/` | Vector store, MMR, multi-query, contextual compression, and Wikipedia retrievers |
| `retrieval-augmented-generation/` | End-to-end RAG: a YouTube-transcript chatbot |

### `langGraph-practice/`

LangGraph experiments — currently a placeholder as this section is being built out.

## Getting started

### 1. Prerequisites

- Python 3.13 (a virtual environment lives in `langChain-practice/env/`)

### 2. Install dependencies

```bash
cd langChain-practice
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the repository root with the keys you need:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

### 4. Run an example

Most scripts are standalone and runnable directly:

```bash
python chains/simple_chain.py
python retrieval-augmented-generation/youtube_chatbot.py
```

For the Streamlit-based prompt UI:

```bash
streamlit run prompts/prompt_ui.py
```

## Key dependencies

LangChain (`langchain`, `langchain-core`, `langchain-community`, `langchain-experimental`), provider integrations (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-huggingface`), vector stores (`chromadb`, `langchain-chroma`, `faiss-cpu`), plus `pydantic`, `python-dotenv`, `streamlit`, `tiktoken`, `pypdf`, `wikipedia`, and `youtube-transcript-api`. See [`requirements.txt`](langChain-practice/requirements.txt) for the full list.

## Status

Actively used for learning — LangChain fundamentals are largely covered, and LangGraph work is in progress.

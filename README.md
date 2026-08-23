# AI-agents

A hands-on learning repository for building AI agents with **LangChain** and **LangGraph**. It works through the core building blocks of LLM applications — chat models, prompts, structured output, chains, runnables, retrievers, vector stores and RAG — and then moves on to graph-based agents: workflows, tools, persistence, human-in-the-loop, subgraphs, short- and long-term memory, and a multi-agent blog writing project.

## Repository layout

```
AI-agents/
├── langChain-practice/     # LangChain fundamentals (Python scripts + notebooks)
└── langGraph-practice/     # LangGraph workflows, agents and memory (notebooks)
```

Each sub-project has its own virtual environment and `requirements.txt`, so they can be installed and run independently.

---

## `langChain-practice/`

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
| `tools/` | Built-in tools, custom tools, toolkits, and tool calling |
| `agents/` | A simple ReAct-style agent |

---

## `langGraph-practice/`

Numbered notebooks meant to be worked through in order. They build up from plain graphs to memory-backed agents.

### Graph fundamentals & workflows

| Notebook | What it covers |
| --- | --- |
| `0_test_installation.ipynb` | Verify the LangGraph install |
| `1_bmi_workflow.ipynb` | First `StateGraph`: state, nodes, edges (no LLM) |
| `2_simple_llm_workflow.ipynb` | Single-node LLM Q&A graph |
| `3_prompt_chaining.ipynb` | Prompt chaining — outline → content → structured scoring |
| `4_batsman_workflow.ipynb` | Parallel nodes fanning into a summary node |
| `5_UPSC_essay_workflow.ipynb` | Parallel evaluators with structured output and reducers |
| `6_quadratic_equation_workflow.ipynb` | Conditional edges and branching on computed state |
| `7_review_reply_workflow.ipynb` | Sentiment routing → diagnosis → tailored reply |
| `8_X_post_generator.ipynb` | Generator / evaluator / optimizer loop with an iteration cap |

### Chat, tools and persistence

| Notebook | What it covers |
| --- | --- |
| `9_basic_chatbot.ipynb` | Message state with `add_messages` |
| `10_persistence.ipynb` | Checkpointers, threads, state history, time travel |
| `11_tools.ipynb` | `bind_tools`, `ToolNode`, `tools_condition` (search, calculator, stock price) |
| `12_mcp.py` | Connecting to an MCP server with `langchain-mcp-adapters` |
| `13_rag.ipynb` | RAG over a PDF inside a graph |
| `14_hitl.ipynb` | Human-in-the-loop with `interrupt` and `Command(resume=...)` |
| `15_subgraphs.ipynb` | Subgraphs with their own state schema |
| `15_subgraph_shared.ipynb` | Subgraphs sharing the parent state schema |

### Short-term memory (STM)

| Notebook | What it covers |
| --- | --- |
| `16_stm_trimming.ipynb` | `trim_messages` to keep history inside a token budget |
| `17_stm_deletion.ipynb` | Dropping old turns with `RemoveMessage` |
| `18_stm_summarization.ipynb` | Rolling conversation summary that replaces old messages |

### Long-term memory (LTM)

| Notebook | What it covers |
| --- | --- |
| `19_ltm_basics.ipynb` | `InMemoryStore`: namespaces, `put` / `get` / `search` |
| `20_ltm_implementation.ipynb` | A chatbot that reads pre-seeded user memories |
| `21_ltm_chat.ipynb` | Chatbot that extracts and writes its own memories, with de-duplication |
| `22_ltm_chat_postgres.ipynb` | Same graph on `PostgresStore` + `PostgresSaver` so memory survives restarts |

### Blog Writing Agent (BWA)

A multi-agent orchestrator/worker project, built up over three notebooks. Outputs land in `articles/`.

| Notebook | What it covers |
| --- | --- |
| `23_bwa_basic.ipynb` | Orchestrator plans sections, `Send` fans out to parallel workers, synthesizer joins them |
| `24_bwa_improved_prompting.ipynb` | Richer plan schema (goal, bullets, target length) and much stronger prompts |
| `25_bwa_research.ipynb` | Adds a research phase — query generation, web search, evidence collected before planning |

---

## Getting started

### 1. Prerequisites

- Python 3.13
- A PostgreSQL database (only for `22_ltm_chat_postgres.ipynb`)

### 2. Install dependencies

For the LangChain track:

```bash
cd langChain-practice
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
```

For the LangGraph track:

```bash
cd langGraph-practice
python -m venv myenv
source myenv/bin/activate      # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the repository root with the keys you need:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token

# Optional — LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true

# Optional — only for the Postgres LTM notebook
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/postgres
```

### 4. Run an example

LangChain scripts are standalone and runnable directly:

```bash
python chains/simple_chain.py
python retrieval-augmented-generation/youtube_chatbot.py
```

For the Streamlit-based prompt UI:

```bash
streamlit run prompts/prompt_ui.py
```

LangGraph work lives in notebooks — open the folder in Jupyter or VS Code and run them in order:

```bash
jupyter lab langGraph-practice
```

## Key dependencies

**LangChain track:** `langchain`, `langchain-core`, `langchain-community`, `langchain-classic`, `langchain-experimental`, provider integrations (`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-huggingface`), vector stores (`chromadb`, `langchain-chroma`, `faiss-cpu`), plus `pydantic`, `python-dotenv`, `streamlit`, `tiktoken`, `pypdf`, `wikipedia`, `youtube-transcript-api`, and `ddgs`.

**LangGraph track:** `langgraph` with `langgraph-checkpoint`, `langgraph-checkpoint-sqlite`, `langgraph-checkpoint-postgres`, `langgraph-prebuilt`, and `langchain-mcp-adapters` / `mcp` for MCP tooling.

Full pinned lists: [`langChain-practice/requirements.txt`](langChain-practice/requirements.txt) and [`langGraph-practice/requirements.txt`](langGraph-practice/requirements.txt).

## Status

Actively used for learning. LangChain fundamentals are covered; the LangGraph track runs through workflows, tools, persistence, HITL, subgraphs, and both memory types, and the Blog Writing Agent is the current project.

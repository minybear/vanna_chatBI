# Vanna 2.0 Project Context (DEEPV.md)

## 1. Project Overview
**Vanna 2.0** is a Python-based AI SQL agent framework that enables "Chat with your Database" functionality. It transforms natural language questions into SQL queries, executes them securely, and visualizes the results.

**Key Features:**
*   **Text-to-SQL:** Uses LLMs to generate accurate SQL for various databases.
*   **User-Aware Security:** Implements permissions and row-level security at the tool level.
*   **Streaming Architecture:** Streams rich UI components (tables, charts) to the frontend via Server-Sent Events (SSE).
*   **Modular Design:** Highly extensible architecture for LLMs, databases, and tools.
*   **Web Component:** Includes a pre-built `<vanna-chat>` web component for easy integration.

## 2. Architecture & Core Components

The project follows a modular, interface-driven architecture located in `src/vanna`.

### Core Modules (`src/vanna/core`)
*   **`Agent` (`agent/agent.py`):** The central brain that orchestrates interactions between the user, LLM, and tools.
*   **`Tool` (`tool/tool.py`):** Base class for all capabilities. Tools are user-aware and permission-gated.
    *   *Examples:* `RunSqlTool`, `VisualizeDataTool`.
*   **`ComponentManager` (`component_manager.py`):** Manages the streaming of UI components (Rich Components) to the client.
*   **`Registry` (`registry.py`):** Handles registration and retrieval of tools and services.

### Integrations (`src/vanna/integrations`)
Pluggable implementations for external services:
*   **LLMs:** `openai`, `anthropic`, `ollama`, `google`, etc. (Implement `LlmService`).
*   **Databases:** `postgres`, `snowflake`, `bigquery`, `sqlite`, etc. (Implement `SqlRunner`).
*   **Vector Stores:** `chromadb`, `pinecone`, `qdrant` (for RAG context).

### Servers (`src/vanna/servers`)
*   **`base`:** Contains the core `ChatHandler` logic for processing requests.
*   **`fastapi` / `flask`:** Framework-specific adapters to expose the chat endpoints (e.g., `/api/vanna/v2/chat_sse`).

### Frontend (`frontends/webcomponent`)
*   **`<vanna-chat>`:** A custom web component built with **Lit** and **Vite**.
*   **Communication:** Consumes the SSE stream from the backend and renders `RichComponents` (Tables, Plotly charts, Markdown).

## 3. Tech Stack

### Backend (Python)
*   **Language:** Python 3.9+
*   **Core Libs:** `pydantic` (Validation), `pandas` (Data manipulation), `plotly` (Visualization), `sqlalchemy` (DB connection).
*   **Web Frameworks:** `fastapi`, `flask` (Supported adapters).
*   **LLM Clients:** `openai`, `anthropic`, `google-generativeai`.

### Frontend (TypeScript)
*   **Framework:** Lit (Web Components).
*   **Build Tool:** Vite.
*   **Visualization:** Plotly.js.
*   **Styling:** CSS Variables (Themable).

## 4. Key Workflows

### Natural Language to Visualization
1.  **User Request:** User sends "Show sales by region" via `<vanna-chat>`.
2.  **Agent Processing:**
    *   `Agent` receives the request and user context.
    *   `Agent` selects relevant tools (e.g., `RunSqlTool`).
3.  **SQL Generation:**
    *   `RunSqlTool` uses the configured `LlmService` to generate SQL.
4.  **Execution & Security:**
    *   `RunSqlTool` checks user permissions.
    *   `SqlRunner` executes the query against the database.
5.  **Visualization:**
    *   Data is saved to a CSV/Parquet cache.
    *   `VisualizeDataTool` generates a Plotly figure config.
6.  **Streaming Response:**
    *   Server streams events: `sql_code` -> `data_table` -> `plotly_chart` -> `summary`.
    *   Frontend renders these components progressively.

## 5. Directory Structure

```text
src/vanna/
├── agents/             # Pre-built agent implementations
├── capabilities/       # Core capabilities (SQL runner, File system)
├── components/         # UI Component definitions (Rich/Simple)
├── core/               # Core framework logic
│   ├── agent/          # Agent orchestration
│   ├── llm/            # LLM service interfaces
│   ├── tool/           # Tool base classes
│   └── ...
├── examples/           # Usage examples and demos
├── integrations/       # Adapters for LLMs, DBs, Vector Stores
├── legacy/             # Vanna 0.x compatibility layer
├── servers/            # Web server adapters (FastAPI, Flask)
└── utils/              # Helper functions
```

## 6. Development & Testing

### Setup
*   **Dependency Management:** `pyproject.toml` (Poetry/Hatch standard).
*   **Install:** `pip install -e .[dev]`

### Testing
*   **Framework:** `pytest`.
*   **Location:** `tests/`.
*   **Run Tests:**
    ```bash
    pytest tests/
    ```
*   **Coverage:** Focus on `test_workflow.py` for end-to-end flows and `test_agents.py` for core logic.

### Frontend Development
*   Navigate to `frontends/webcomponent`.
*   Install: `npm install`.
*   Dev Server: `npm run dev`.
*   Build: `npm run build`.

## 7. Conventions & Patterns

*   **Typing:** Strong typing with Python `typing` and `Pydantic` models is mandatory.
*   **Async/Await:** The core `Agent` and `Tool` execution paths are asynchronous (`async def`).
*   **Dependency Injection:** Services (LLM, DB) are injected into the Agent/Tools, promoting testability.
*   **Rich Components:** UI elements are defined as Pydantic models in `src/vanna/components` and serialized to JSON for the frontend.

## DeepV Code Added Memories
- DEEPV.md generated by /init command on 2025-12-05 12:00:00
- User uses GLM-4.6 model with endpoint https://open.bigmodel.cn/api/anthropic and API key 89cd3555805348baa2b9df35dfec0e02.BeP5cDZydHmjxEwW
- Database connection info: jdbc:mariadb://10.18.50.25:3306/, username: root, password: redtea.123

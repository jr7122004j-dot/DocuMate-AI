# Changelog

All notable changes to DocuMate-AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-28

### Added
- **Decoupled Service Layer (`DocuMateEngine`):** UI-agnostic orchestration of document ingestion,
  vector retrieval, semantic caching, and streaming question answering.
- **Typed Configuration (`Settings`):** Unified configuration powered by `pydantic-settings` with
  `DOCUMATE_` environment variable overrides, input validation, and relative path resolution.
- **Structured Error Hierarchy (`errors.py`):** Typed domain exceptions including `DocuMateError`,
  `EngineNotReadyError`, `OllamaUnavailableError`, `ModelNotFoundError`, and
  `DocumentProcessingError`.
- **Centralized Logging (`logging_config.py`):** Configurable log levels with clean console
  formatting.
- **Ollama Health Diagnostics (`check_ollama_health`):** Real-time server and model availability
  checks with actionable CLI pull command guidance in the UI.
- **Pinpoint Citations (`schemas.py:Citation`):** Structured citation transfer objects holding
  1-based page numbers, document sources, and contextual text snippets.
- **Deterministic Token Streaming (`stream_handler.py`):** `TokenSink` protocol with
  `StreamlitTokenSink` (throttled rendering + typing cursor), `CollectingSink` (in-memory testing),
  and `NullSink`.
- **Modular Streamlit UI Components (`src/documate_ai/ui/`):**
  - `sidebar.py`: Ollama health banner, multi-PDF uploader with `st.status` progress, loaded
    documents list, semantic cache controls with dynamic bounds, Clear Chat, and Reset All.
  - `chat.py`: Pre-ingest guidance banner, history replay with collapsible Sources expanders,
    cache hit indicators (`⚡ Cached answer`), and execution timings breakdown.
  - `resources.py`: Cached resource loaders for embeddings, language models, and cache managers.
- **GitHub Actions CI Pipeline (`.github/workflows/ci.yml`):** Automated test matrix across
  Python 3.10, 3.11, 3.12, and 3.13 with mypy type checking and coverage enforcement.
- **Comprehensive QA Documentation (`docs/QA.md`):** Complete feature-parity audit and Windows
  edge-case analysis.
- **Live Ollama Integration Tests (`test/integration/test_live_ollama.py`):** End-to-end tests
  marked `@pytest.mark.ollama` verifying ingestion, QA, citations, and cache hits.

### Changed
- **LangChain Expression Language (LCEL):** Replaced legacy `ConversationalRetrievalChain` with
  modern LCEL pipelines using isolated history snapshots to prevent context leakage.
- **Cache Fingerprinting:** Upgraded cache keys to SHA-256 document-content hashing and
  query vector cosine similarity matrix matching.
- **Streamlit Modernisation:** Replaced deprecated `use_container_width=True` with modern
  `width="stretch"` button styling.
- **Application Entry Point:** Refactored `app.py` into a lightweight coordinator (`main()`).
- **Dynamic Versioning:** Single-sourced version `1.0.0` from `src/documate_ai/__init__.py` using
  `hatchling`.

### Fixed
- Fixed 0-based page index bug in PDF loading; citations now consistently display 1-based page
  numbers.
- Fixed condense-question prompt leakage where intermediate rephrased query tokens streamed into
  the chat output.
- Fixed multi-question history pollution where sub-question turns leaked into the session history.
- Fixed ChromaDB crashes on empty or image-only scanned PDFs by adding empty-text validation.
- Fixed Windows OpenTelemetry / gRPC telemetry DLL blocks by explicitly setting
  `ANONYMIZED_TELEMETRY="False"`.
- Fixed orphaned Chroma vector store directories remaining on disk after "Reset All" actions.

### Security
- Prevented raw Python tracebacks and sensitive filesystem paths from leaking into the user
  interface.
- Bound all internal services to localhost (`127.0.0.1`) by default.

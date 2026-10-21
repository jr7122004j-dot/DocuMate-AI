# DocuMate-AI
A secure, on-device RAG chatbot for multi-PDF analysis built with Python, LangChain, and Ollama, featuring semantic caching and precise source citations.


• Secure, On-Device Processing: Runs entirely locally using Ollama, ensuring complete data privacy without API keys or external dependencies.
• Robust Multi-PDF Support: Processes multiple PDF files (up to 500 pages) with smart chunking and automatic source deduplication for consistent context.
• Semantic Caching: Implements a VectorStore-based cache that retrieves relevant context from previous queries, reducing LLM usage and improving response time.
• Pinpointing Citations: Matches answers to source chunks using embedding similarity and cosine similarity, providing precise reference tracking for every response.
• Flexible PDF Interaction: Supports both single-PDF mode (uploaded via UI) and multi-PDF mode (scans entire project folder), adaptable to user needs.

## Prerequisites & Development Environment

- **Python Version**: `>=3.10, <3.14` (Recommended: **Python 3.12** or **3.13**).
  - *Note*: ChromaDB and its native dependencies (`chroma-hnswlib`, `tokenizers`) currently lack pre-built binary wheels for Python 3.14 on Windows and will fail to install from source without MSVC C++ Build Tools and a Rust toolchain. Use a Python 3.12 or 3.13 virtual environment for full runtime installation.
- **Ollama**: Local Ollama server (`http://localhost:11434`) with `phi4-mini` and `all-minilm:l6-v2` pulled.

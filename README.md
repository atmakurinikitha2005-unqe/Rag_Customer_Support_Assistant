# RAG Customer Support Assistant

A dynamic, user-driven RAG system using LangGraph and ChromaDB.

## Features
- **Session-Based Isolation**: Each Streamlit session gets its own isolated ChromaDB collection.
- **Dynamic Ingestion**: Upload PDFs directly from the UI to build the knowledge base on the fly.
- **LangGraph Routing**: Analyzes queries and routes them intelligently (answer, clarify, fallback, escalate).
- **Human-In-The-Loop (HITL)**: Simulates escalation to a human reviewer when sensitive questions are asked or confident answers cannot be found.
- **Local Embeddings**: Defaults to `all-MiniLM-L6-v2` via HuggingFace for easy local testing.

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure your API keys (especially `GROQ_API_KEY` for the LLM).

## Running the App
Run the Streamlit application:
```bash
streamlit run app/ui/streamlit_app.py
```


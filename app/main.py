from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from app.graph.builder import build_graph
from app.ingestion import VectorStore

load_dotenv()


class SupportAssistant:
    """Main class used by Streamlit UI."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4()).replace("-", "_")
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
        self.upload_dir.mkdir(exist_ok=True)

        self.vector_store = VectorStore(
            collection_name=f"support_{self.session_id}",
            persist_dir=os.getenv("CHROMA_DB_DIR", "./chroma_db"),
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        )
        self.graph = build_graph()
        self.llm = self._load_llm()

    def _load_llm(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"')
        if not api_key or api_key == "your_groq_api_key_here":
            return None
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)

    def ingest_pdf(self, uploaded_file) -> int:
        file_path = self.upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return self.vector_store.add_pdf(file_path, uploaded_file.name)

    def ask(self, question: str) -> str:
        context = self.vector_store.search(question, k=4)

        if self.llm and context:
            prompt = (
                "You are a helpful customer support assistant. Answer only from the given context.\n\n"
                f"Context:\n{' '.join(context)}\n\nQuestion: {question}"
            )
            return self.llm.invoke(prompt).content

        result = self.graph.invoke({"question": question, "context": context, "answer": ""})
        return result["answer"]

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable, List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader


def read_pdf_text(file_path: str | Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks for retrieval."""
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = start + chunk_size
        chunks.append(clean[start:end])
        start = max(end - overlap, end)
    return chunks


class VectorStore:
    """Small ChromaDB wrapper used by the Streamlit app."""

    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_pdf(self, file_path: str | Path, source_name: str) -> int:
        text = read_pdf_text(file_path)
        chunks = split_text(text)
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source_name, "chunk": i} for i in range(len(chunks))]
        self.collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[str]:
        if not query.strip():
            return []
        results = self.collection.query(query_texts=[query], n_results=k)
        return results.get("documents", [[]])[0]

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

# Allows running with: streamlit run app/ui/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import SupportAssistant


st.set_page_config(page_title="RAG Customer Support Assistant", page_icon="🤖", layout="wide")
st.title("🤖 RAG Customer Support Assistant")
st.write("Upload a PDF, then ask questions from that document.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()).replace("-", "_")

if "assistant" not in st.session_state:
    st.session_state.assistant = SupportAssistant(st.session_state.session_id)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Ingest PDF"):
        with st.spinner("Reading and indexing PDF..."):
            chunks = st.session_state.assistant.ingest_pdf(uploaded_file)
        if chunks:
            st.success(f"PDF indexed successfully. Created {chunks} text chunks.")
        else:
            st.warning("No readable text found in this PDF.")

question = st.text_input("Ask a question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching document..."):
            answer = st.session_state.assistant.ask(question)
        st.subheader("Answer")
        st.write(answer)

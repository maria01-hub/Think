import streamlit as st
import os
import tempfile
from rag_engine import (
    load_and_split_pdf,
    create_vector_store,
    build_qa_chain,
    answer_question
)

st.set_page_config(page_title="Think_Chatbot", page_icon="🧠")
st.title("🧠 Think")
st.caption("Upload a PDF and ask questions about it — powered by RAG")

# Session state to persist chain across reruns
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Reading and indexing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_split_pdf(tmp_path)
            vector_store = create_vector_store(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_store)
            st.session_state.messages = []
            os.unlink(tmp_path)

        st.success(f"Document processed! ({len(chunks)} chunks indexed)")

# --- Main: Chat ---
if st.session_state.qa_chain is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    question = st.chat_input("Ask something about your document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(st.session_state.qa_chain, question)
                st.write(answer)

                with st.expander("📚 View sources"):
                    for i, doc in enumerate(sources):
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Source {i+1} (page {page}):**")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
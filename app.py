import streamlit as st
import os
import tempfile
from rag_engine import (
    load_and_split_pdf,
    create_vector_store,
    build_qa_chain,
    answer_question,
    init_history_db,
    save_to_history,
    get_history
)

st.set_page_config(page_title="Think_Chatbot", page_icon="🧠")

# --- Theme definitions ---
THEMES = {
    "Light": {"bg": "#FFFFFF", "sidebar_bg": "#F5F3FF", "text": "#1E1B2E", "primary": "#7C3AED"},
    "Dark": {"bg": "#000000", "sidebar_bg": "#0A0A0A", "text": "#FFFFFF", "primary": "#FFFFFF"},
}

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Dark"
if "custom_bg" not in st.session_state:
    st.session_state.custom_bg = "#000000"
if "custom_text" not in st.session_state:
    st.session_state.custom_text = "#FFFFFF"
if "custom_primary" not in st.session_state:
    st.session_state.custom_primary = "#7C3AED"

# --- Sidebar: theme picker (placed first so it applies before rendering the rest) ---
with st.sidebar:
    st.header("🎨 Appearance")
    theme_choice = st.selectbox(
        "Theme",
        ["Light", "Dark", "Custom"],
        index=["Light", "Dark", "Custom"].index(st.session_state.theme_choice)
    )
    st.session_state.theme_choice = theme_choice

    if theme_choice == "Custom":
        st.session_state.custom_bg = st.color_picker("Background color", st.session_state.custom_bg)
        st.session_state.custom_text = st.color_picker("Text color", st.session_state.custom_text)
        st.session_state.custom_primary = st.color_picker("Accent color", st.session_state.custom_primary)
        active = {
            "bg": st.session_state.custom_bg,
            "sidebar_bg": st.session_state.custom_bg,
            "text": st.session_state.custom_text,
            "primary": st.session_state.custom_primary,
        }
    else:
        active = THEMES[theme_choice]

    st.divider()

# --- Inject CSS for the chosen theme ---
st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Anton&family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp, [class*="css"] {{
        font-family: 'Poppins', sans-serif !important;
    }}
    .stApp {{
        background-color: {active['bg']};
        color: {active['text']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {active['sidebar_bg']};
    }}
    .stButton>button {{
        background-color: {active['primary']};
        color: {active['bg']};
        border: none;
        font-weight: 600;
        font-family: 'Poppins', sans-serif !important;
    }}
    h1 {{
        font-family: 'Anton', sans-serif !important;
        letter-spacing: 1px;
    }}
    h1, h2, h3, p, span, label, .stMarkdown {{
        color: {active['text']} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: {active['text']} !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 500;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {{
        color: {active['text']} !important;
        font-family: 'Poppins', sans-serif !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Think")
st.caption("Upload a PDF and ask questions about it — powered by RAG")

# Make sure the history database exists
init_history_db()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = None

# --- Sidebar: Upload + History ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Reading and indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_split_pdf(tmp_path)
            vector_store = create_vector_store(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_store)
            st.session_state.messages = []
            st.session_state.current_doc_name = uploaded_file.name
            os.unlink(tmp_path)

        st.success(f"Document processed! ({len(chunks)} chunks indexed)")

    st.divider()
    st.header("📜 Past Questions")
    history = get_history()

    if not history:
        st.caption("No past questions yet.")
    else:
        for doc_name, question, answer, timestamp in history[:15]:
            with st.expander(f"{question[:40]}..."):
                st.caption(f"📄 {doc_name} · {timestamp}")
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {answer}")

# --- Main: Chat ---
if st.session_state.qa_chain is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

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

                save_to_history(st.session_state.current_doc_name, question, answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
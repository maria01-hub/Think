import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()


def load_and_split_pdf(pdf_path):
    """Load a PDF and split it into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, persist_directory="chroma_db"):
    """Embed chunks and store them in ChromaDB."""
    if not chunks:
        raise ValueError("No readable text found in this PDF. It may be a scanned/image-only document.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


def load_existing_vector_store(persist_directory="chroma_db"):
    """Load a previously saved vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


def build_qa_chain(vector_store):
    """Build the retrieval + generation chain."""

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt_template = """You are a helpful assistant answering questions based ONLY on the provided context.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer clearly and concisely, citing which part of the document supports your answer."""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def answer_question(qa_chain, question):
    """Run a question through the chain and return answer + sources."""
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources


# --- History database functions ---

def init_history_db():
    """Create the history table if it doesn't exist yet."""
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_to_history(doc_name, question, answer):
    """Save one question-answer pair to the history database."""
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (doc_name, question, answer, timestamp) VALUES (?, ?, ?, ?)",
        (doc_name, question, answer, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )
    conn.commit()
    conn.close()


def get_history():
    """Retrieve all past question-answer pairs, most recent first."""
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT doc_name, question, answer, timestamp FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_uploaded_documents():
    """Get a list of unique document names that have been uploaded before."""
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT doc_name FROM history ORDER BY doc_name")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]
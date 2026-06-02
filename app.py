import streamlit as st
import tempfile

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# ==================================================
# EMBEDDINGS (cached for performance)
# ==================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# ==================================================
# LLM (Gemini / RAG generation)
# ==================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
)

# ==================================================
# BUILD VECTOR DB
# ==================================================

def build_db(pdf_paths):

    all_docs = []

    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return db

# ==================================================
# UI HEADER
# ==================================================

st.title("🧠 Medical Research Assistant (RAG)")
st.write("Upload medical research PDFs and ask questions.")

# ==================================================
# SESSION STATE
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==================================================
# PDF UPLOAD
# ==================================================

uploaded_files = st.file_uploader(
    "Upload Medical PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ==================================================
# PROCESS PDFs (SAFE TEMP FILE HANDLING)
# ==================================================

if uploaded_files and "db_ready" not in st.session_state:

    pdf_paths = []

    for file in uploaded_files:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            pdf_paths.append(tmp.name)

    st.success("PDFs uploaded successfully!")

    with st.spinner("Building knowledge base..."):
        db = build_db(pdf_paths)

        st.session_state["db"] = db
        st.session_state["db_ready"] = True

    st.success("Knowledge Base Ready!")

# ==================================================
# SUMMARY
# ==================================================

if st.button("📄 Summarize Uploaded Paper"):

    if "db" not in st.session_state:
        st.warning("Upload PDF first.")
    else:
        db = st.session_state["db"]

        docs = db.similarity_search("summary of the paper", k=5)

        summary_text = "\n\n".join(
            [doc.page_content[:500] for doc in docs]
        )

        st.subheader("Paper Summary")
        st.write(summary_text)

# ==================================================
# QUESTION INPUT
# ==================================================

question = st.text_input("Ask a medical question:")

# ==================================================
# QUESTION ANSWERING (REAL RAG)
# ==================================================

if question:

    if "db" not in st.session_state:
        st.warning("Please upload a PDF first.")
        st.stop()

    db = st.session_state["db"]

    results = db.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in results])

    # ==================================================
    # CONTEXT VIEWER
    # ==================================================

    with st.expander("📚 Retrieved Context"):
        st.write(context)

    # ==================================================
    # LLM RESPONSE (REAL AI ANSWER)
    # ==================================================

    prompt = f"""
You are a medical research assistant.

Use the context below to answer the question accurately.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)
    answer = response.content

    st.subheader("💡 Answer")
    st.write(answer)

    # ==================================================
    # SOURCES
    # ==================================================

    st.subheader("📖 Sources")

    for i, doc in enumerate(results):
        page = doc.metadata.get("page", "Unknown")
        st.write(f"Source {i+1} - Page {page}")

    # ==================================================
    # DOWNLOAD ANSWER
    # ==================================================

    st.download_button(
        label="⬇ Download Answer",
        data=answer,
        file_name="medical_answer.txt",
        mime="text/plain"
    )

    # ==================================================
    # CHAT HISTORY
    # ==================================================

    st.session_state.history.append({
        "question": question,
        "answer": answer
    })

# ==================================================
# CHAT HISTORY DISPLAY
# ==================================================

if st.session_state.history:

    st.subheader("📝 Previous Questions")

    for item in reversed(st.session_state.history):

        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer'][:300]}...")
        st.divider()
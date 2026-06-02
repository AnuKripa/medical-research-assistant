# 🧠 Medical Research Assistant (RAG Chatbot)

An AI-powered **Medical Research Assistant** built using Retrieval-Augmented Generation (RAG).  
This system allows users to upload research PDFs and ask context-aware medical questions, which are answered using semantic search + LLM reasoning.

---

## 🚀 Features

- 📄 Upload and process medical research PDFs
- 🔍 Chunking and embedding of document content
- 🧠 Vector-based semantic search (FAISS / similar DB)
- 💬 AI-powered question answering using Gemini / LLM
- ⚡ Fast retrieval of relevant context from documents
- 🖥️ Simple Streamlit web interface

---

## 🏗️ System Architecture

1. **PDF Ingestion**
   - Reads and extracts text from research papers

2. **Chunking**
   - Splits documents into smaller meaningful sections

3. **Embedding Generation**
   - Converts text chunks into vector embeddings

4. **Vector Storage**
   - Stores embeddings for similarity search

5. **Retrieval**
   - Finds most relevant chunks based on user query

6. **LLM Response**
   - Generates final answer using retrieved context

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit
- FAISS / Vector Database
- Google Gemini API / LLM
- LangChain (if used)
- PyPDF / PDF parsing libraries

---

## 📂 Project Structure
rag-chatbot/
│── app.py # Streamlit UI
│── rag_chat.py # Core RAG pipeline
│── read_pdf.py # PDF loader
│── chunk_pdf.py # Text chunking
│── embeddings.py # Embedding generation
│── store_embeddings.py # Store vectors
│── query_db.py # Retrieval logic
│── test_gemini.py # API testing
│── .gitignore



---

## ⚙️ Installation & Setup

### 1. Clone the repository
bash
git clone https://github.com/AnuKripa/medical-research-assistant.git
cd medical-research-assistant

### 2. Create virtual environment
bash
python -m venv venv
venv\Scripts\activate   # Windows

### 3. install Dependencies
bash
pip install -r requirements.txt

### 4.Run the app
bash
streamlit run app.py

**Environment Variables**

Create a .env file and add your API keys:

GEMINI_API_KEY=your_api_key_here

**Example Use Case**
Upload a medical research paper (PDF)

Ask:

"What are the findings of this study?"

Get AI-generated answers based only on the document content.


**Future Improvements**
Improve chunking strategy (semantic chunking)
Add multi-document QA
Add citation-based answers (page-level references)
Deploy on Streamlit Cloud / HuggingFace Spaces
Add chat history memory

### 👍 If you want to make it even better
I can also help you add:
- 🔥 badges (Python, Streamlit, etc.)
- 📸 screenshots section
- 🧾 architecture diagram (very impressive for resume)
- 🚀 deployment link section (Streamlit Cloud)

Just tell me 👍 

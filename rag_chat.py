from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# 👇 ADD IT HERE
print("Total documents in DB:", db._collection.count())

question = input("Ask: ")

results = db.similarity_search(question, k=3)

context = "\n".join([doc.page_content for doc in results])

print("\nRetrieved Context:\n")
print(context)
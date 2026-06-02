from sentence_transformers import SentenceTransformer
import chromadb

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("brain_tumor_paper")

# User question
query = "What deep learning model was used?"

# Convert question to embedding
query_embedding = model.encode(query).tolist()

# Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nQuestion:")
print(query)

print("\nTop Results:\n")

for doc in results["documents"][0]:
    print(doc)
    print("-" * 50)
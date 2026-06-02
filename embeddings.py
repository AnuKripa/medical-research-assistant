from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Brain tumor detection using MRI"

embedding = model.encode(text)

print("Embedding length:", len(embedding))
print(embedding[:10])
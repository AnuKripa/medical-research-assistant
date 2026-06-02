from pypdf import PdfReader

reader = PdfReader("data/paper.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text() or ""

chunk_size = 500

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

print("Number of chunks:", len(chunks))
print("\nFirst Chunk:\n")
print(chunks[0])
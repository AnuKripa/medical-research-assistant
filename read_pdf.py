from pypdf import PdfReader

reader = PdfReader("data/paper.pdf")

print("Number of pages:", len(reader.pages))

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

print(text[:2000])
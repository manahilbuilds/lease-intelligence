from pypdf import PdfReader
import ollama


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


# Read PDF
reader = PdfReader("data/leases/lease1.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

# Create chunks
chunks = chunk_text(text)

print(f"Created {len(chunks)} chunks")

# Create embeddings for ALL chunks
embeddings = []

for chunk in chunks:
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    embeddings.append(response["embeddings"][0])

print(f"\nCreated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} numbers")
from pypdf import PdfReader
import ollama
import chromadb
import os


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# =====================================================
# READ ALL PDFS
# =====================================================

all_chunks = []

for filename in os.listdir("data/leases"):

    if filename.endswith(".pdf"):

        path = os.path.join("data/leases", filename)

        print(f"Reading {filename}")

        reader = PdfReader(path)

        for page_number, page in enumerate(reader.pages, start=1):

            page_text = page.extract_text()

            if not page_text:
                continue

            chunks = chunk_text(page_text)

            for chunk in chunks:

                all_chunks.append(
                    {
                        "text": chunk,
                        "source": filename,
                        "page": page_number
                    }
                )

print(f"\nCreated {len(all_chunks)} chunks")


# =====================================================
# CREATE CHROMADB (Persistent)
# =====================================================

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


# =====================================================
# INDEX DOCUMENTS (ONLY FIRST TIME)
# =====================================================

if collection.count() == 0:

    print("\nNo existing database found.")
    print("Creating embeddings and indexing documents...\n")

    embeddings = []

    for item in all_chunks:

        response = ollama.embed(
            model="nomic-embed-text",
            input=item["text"]
        )

        embeddings.append(response["embeddings"][0])

    print(f"Created {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} numbers")

    for i, (item, embedding) in enumerate(zip(all_chunks, embeddings)):

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[item["text"]],
            metadatas=[
                {
                    "source": item["source"],
                    "page": item["page"]
                }
            ]
        )

    print(f"\nStored {collection.count()} chunks in ChromaDB")

else:

    print(f"\nLoaded existing database with {collection.count()} chunks")


# =====================================================
# QUESTION ANSWERING
# =====================================================

question = input("\nAsk a question: ")

question_embedding = ollama.embed(
    model="nomic-embed-text",
    input=question
)["embeddings"][0]

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]
retrieved_sources = results["metadatas"][0]

print("\nRetrieved Context:\n")

for chunk, source in zip(retrieved_chunks, retrieved_sources):

    print(f"Source: {source['source']} | Page: {source['page']}")
    print(chunk)
    print("-" * 80)

context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are a lease analysis assistant.

Answer ONLY using the provided context.

If the answer isn't in the context, say:

"I couldn't find that information in the lease."

When you answer, mention which document and page(s) the answer came from if possible.

Context:
{context}

Question:
{question}
"""

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n==========================")
print("ANSWER")
print("==========================\n")

print(response["message"]["content"])
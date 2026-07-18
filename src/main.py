from pypdf import PdfReader
import ollama
import chromadb
import os


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

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

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        chunks = chunk_text(text)

        for chunk in chunks:

            all_chunks.append(
                {
                    "text": chunk,
                    "source": filename
                }
            )

print(f"\nCreated {len(all_chunks)} chunks")


# =====================================================
# CREATE EMBEDDINGS
# =====================================================

embeddings = []

for item in all_chunks:

    response = ollama.embed(
        model="nomic-embed-text",
        input=item["text"]
    )

    embeddings.append(response["embeddings"][0])

print(f"\nCreated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} numbers")


# =====================================================
# CREATE CHROMADB
# =====================================================

client = chromadb.Client()

try:
    client.delete_collection("documents")
except:
    pass

collection = client.get_or_create_collection(
    name="documents"
)


# =====================================================
# STORE IN CHROMA
# =====================================================

for i, (item, embedding) in enumerate(zip(all_chunks, embeddings)):

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[item["text"]],
        metadatas=[
            {
                "source": item["source"]
            }
        ]
    )

print(f"\nStored {collection.count()} chunks")


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

    print(f"Source: {source['source']}")
    print(chunk)
    print("-" * 80)


context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are a lease analysis assistant.

Answer ONLY using the provided context.

If the answer isn't in the context, say:

"I couldn't find that information in the lease."

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
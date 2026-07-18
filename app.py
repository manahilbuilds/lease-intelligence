import streamlit as st
from pypdf import PdfReader
import ollama
import chromadb
import os
import tempfile


# -----------------------------
# Chunk text
# -----------------------------
def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# -----------------------------
# ChromaDB
# -----------------------------
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("documents")


st.set_page_config(page_title="Lease Intelligence", page_icon="📄")

st.title("📄 Lease Intelligence")
st.write("Upload one or more lease PDFs and ask questions.")


# =====================================================
# Upload PDFs
# =====================================================

uploaded_files = st.file_uploader(
    "Upload lease PDFs",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files:

    if st.button("Index Documents"):

        try:
            client.delete_collection("documents")
        except:
            pass

        collection = client.get_or_create_collection("documents")

        all_chunks = []

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

                tmp.write(uploaded_file.read())

                pdf_path = tmp.name

            reader = PdfReader(pdf_path)

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if text:

                    chunks = chunk_text(text)

                    for chunk in chunks:

                        all_chunks.append(
                            {
                                "text": chunk,
                                "source": uploaded_file.name,
                                "page": page_number
                            }
                        )

            os.remove(pdf_path)

        progress = st.progress(0)

        embeddings = []

        for i, item in enumerate(all_chunks):

            embedding = ollama.embed(
                model="nomic-embed-text",
                input=item["text"]
            )["embeddings"][0]

            embeddings.append(embedding)

            progress.progress((i + 1) / len(all_chunks))

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

        st.success(f"Indexed {collection.count()} chunks.")

st.success(f"Indexed {collection.count()} chunks.")

st.write("### Indexed Files")

for file in uploaded_files:
    st.write(f"✅ {file.name}")
    
# =====================================================
# Ask Question
# =====================================================

question = st.text_input("Ask a question")


if st.button("Get Answer") and question:

    question_embedding = ollama.embed(
        model="nomic-embed-text",
        input=question
    )["embeddings"][0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    chunks = results["documents"][0]
    metadata = results["metadatas"][0]

    context = "\n\n".join(chunks)

    prompt = f"""
You are a lease analysis assistant.

Answer ONLY using the provided context.

If the answer is not found, say:

"I couldn't find that information."

At the end ALWAYS cite the source filename and page.

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

    st.subheader("Answer")

    st.write(response["message"]["content"])

    st.subheader("Retrieved Chunks")

    for chunk, meta in zip(chunks, metadata):

        st.markdown(
            f"**{meta['source']} — Page {meta['page']}**"
        )

        st.write(chunk)

        st.divider()
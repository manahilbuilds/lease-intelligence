# 📄 Lease Intelligence

An AI-powered Retrieval-Augmented Generation (RAG) application for analyzing lease agreements using local Large Language Models (LLMs).

The system indexes one or more lease PDFs, stores vector embeddings in ChromaDB, retrieves the most relevant sections using semantic search, and generates grounded answers with document and page citations.

---

## Features

- 📑 Index multiple PDF lease agreements
- 🔍 Semantic search using vector embeddings
- 🧠 Local LLM-powered question answering (Ollama + Qwen2.5)
- 📚 ChromaDB vector database
- 📄 Source document and page citations
- 💻 Command-line interface
- 🌐 Streamlit web interface

---

## Tech Stack

- Python
- Ollama
- Qwen2.5 7B
- Nomic Embed Text
- ChromaDB
- Streamlit
- PyPDF

---

## Project Architecture

```
                PDF Leases
                     │
                     ▼
               PDF Extraction
                 (PyPDF)
                     │
                     ▼
              Text Chunking
                     │
                     ▼
      Embeddings (Nomic Embed)
                     │
                     ▼
          ChromaDB Vector Store
                     │
     User Question ──► Embed Query
                     │
                     ▼
          Semantic Retrieval
                     │
                     ▼
      Retrieved Context + Prompt
                     │
                     ▼
         Qwen2.5 (Local LLM)
                     │
                     ▼
        Answer + Source Citation
```

---

## Project Structure

```
lease-intelligence/
│
├── app.py                 # Streamlit application
├── src/
│   └── main.py            # CLI application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/manahilbuilds/lease-intelligence.git
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download Ollama from

https://ollama.com

Pull the required models

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

---

## Running the CLI

```bash
python src/main.py
```

---

## Running the Streamlit App

```bash
streamlit run app.py
```

---


## Data

The `data/` directory is intentionally excluded from this repository via `.gitignore`.

To use the CLI application:

1. Create the following folder:

```text
data/leases/
```

2. Place your own lease PDF files inside it.

Example:

```text
data/
└── leases/
    ├── lease1.pdf
    ├── lease2.pdf
```

Alternatively, you can use the Streamlit application to upload lease PDFs directly without creating the folder.

## Example Questions

- When does the lease expire?
- What is the monthly rent?
- Is there a renewal option?
- Who is responsible for utilities?
- What happens if the tenant terminates early?

---

## Current Limitations

- Works best with digital (text-based) PDFs
- OCR is not supported
- Uses local models only
- Basic semantic retrieval (no reranking)

---

## Future Improvements

- Hybrid search (keyword + vector)
- Reranking
- Structured lease extraction
- LangGraph agent workflow
- Docker deployment
- Cloud deployment

---

## What I Learned

Through this project I implemented an end-to-end Retrieval-Augmented Generation (RAG) pipeline and gained hands-on experience with:

- Document chunking strategies
- Embedding models
- Vector databases
- Semantic search
- Prompt grounding
- Local LLM inference
- Building AI applications with Streamlit

---

## Author

**Manahil Noor**

GitHub:
https://github.com/manahilbuilds

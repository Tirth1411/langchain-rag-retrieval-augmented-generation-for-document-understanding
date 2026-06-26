# LangChain RAG — Retrieval-Augmented Generation for Document Understanding 📚🤖

> Two end-to-end **Retrieval-Augmented Generation (RAG)** pipelines that let you ask natural-language
> questions about a PDF and get grounded answers — one built on **OpenAI**, one fully **open-source**
> with Hugging Face.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-009999?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)

## Overview

RAG combines **document retrieval** with a **language model** so answers are grounded in your own
content instead of the model's memory. This project demonstrates the full pattern over a PDF
(*Andrew Ng's Deep Learning notes*): extract text → chunk it → embed the chunks → store them in a
vector index → retrieve the most relevant chunks for a question → generate an answer.

It is implemented **two ways** so you can compare a hosted-API approach with a self-hosted one:

| Notebook | Embeddings | LLM | When to use |
|----------|-----------|-----|-------------|
| `RAG_OpenAI.ipynb` | `OpenAIEmbeddings` | OpenAI LLM via `RetrievalQA` | Highest answer quality, needs an API key |
| `RAG_Transformer.ipynb` | `SentenceTransformerEmbeddings` | Hugging Face `pipeline` (local) | Free / offline, no external API |

## Pipeline

1. **Load & extract** — read the PDF with `PyPDF2`.
2. **Chunk** — split text into overlapping chunks with LangChain's `RecursiveCharacterTextSplitter`.
3. **Embed** — turn chunks into vectors (OpenAI or SentenceTransformers).
4. **Index** — store and search vectors with **FAISS**.
5. **Retrieve + generate** — `RetrievalQA` / `load_qa_chain` fetches the top chunks and feeds them
   to the LLM to answer the question.
6. **Visualize (Transformer notebook)** — project embeddings down to 2-D with **PCA** and **t-SNE**
   (`matplotlib`) to see how chunks cluster in vector space.

## Tech Stack

`Python` · `LangChain` · `FAISS` · `OpenAI` · `Hugging Face Transformers` · `sentence-transformers` · `PyPDF2` · `scikit-learn (PCA/t-SNE)` · `Matplotlib`

## How to Run

```bash
git clone https://github.com/Tirth1411/langchain-rag-retrieval-augmented-generation-for-document-understanding.git
cd langchain-rag-retrieval-augmented-generation-for-document-understanding
pip install langchain faiss-cpu openai sentence-transformers transformers PyPDF2 scikit-learn matplotlib

# OpenAI version needs a key:
export OPENAI_API_KEY="sk-..."
jupyter notebook RAG_OpenAI.ipynb        # OpenAI-powered
jupyter notebook RAG_Transformer.ipynb   # fully open-source
```

> **Note:** OpenAI deprecated `text-davinci-003`; if you run the OpenAI notebook, point it at a
> current model (e.g. `gpt-4o-mini`). The Transformer notebook runs without any API key.

## Repository Structure

```
RAG_OpenAI.ipynb        # RAG with OpenAI embeddings + LLM
RAG_Transformer.ipynb   # RAG with SentenceTransformers + Hugging Face, plus embedding visualization
data/                   # source PDF (Deep Learning notes)
```

## What this demonstrates

- The complete RAG architecture (chunk → embed → index → retrieve → generate) end-to-end.
- A practical comparison between **hosted (OpenAI)** and **open-source (Hugging Face)** stacks.
- Working with vector stores (FAISS) and inspecting the embedding space.

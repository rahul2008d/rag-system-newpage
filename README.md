# Chat With Your Docs (RAG System)

🚀 **[Live Demo](http://34.245.107.59:8501)**  

---
A containerised **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents (PDF/TXT), index them into a vector store, and ask questions grounded strictly in the uploaded content.

The system is intentionally designed to be **simple, explainable, and production-ready**, prioritising engineering clarity and realistic trade-offs over over-engineering.

---

## 🚀 Features

- Upload **PDF / TXT** documents
- Automatic **chunking, embedding, and vector indexing**
- Context-aware Q&A using **OpenAI + FAISS**
- Guardrails to reduce hallucinations (answers only from retrieved context)
- Simple **Streamlit UI**
- Fully **containerised** and deployed on **AWS ECS Fargate**
- Clean engineering standards: linting, formatting, Docker, pre-commit hooks

---

## 🧠 Architecture Overview

```
User
 └── Streamlit UI
      ├── Upload documents
      ├── Ask questions
      │
      ▼
RAG System (LangChain-based)
 ├── Document Loaders (PDF / TXT)
 ├── Text Chunking (overlapping chunks)
 ├── OpenAI Embeddings
 ├── FAISS Vector Store (local)
 ├── Retriever (Top-K similarity)
 └── OpenAI Chat Completion
```


### Design Principles
- **Retrieval before generation**
- **No answer without relevant context**
- **Stateless UI, stateful vector store**
- **Simple first, extensible later**

---

## 🏗️ Tech Stack

| Area | Technology | Reason |
|---|---|---|
| UI | Streamlit | Rapid development, minimal boilerplate |
| RAG Framework | LangChain | Reduces glue code, clean abstractions |
| Embeddings | OpenAI | High-quality semantic representations |
| Vector Store | FAISS | Lightweight, fast, local |
| LLM | OpenAI Chat Models | Reliable, controllable |
| Containerisation | Docker | Reproducible builds |
| Orchestration | AWS ECS Fargate | Serverless containers, low ops |
| Linting | Ruff, Pylint | Consistent code quality |
| CI Hygiene | pre-commit | Catch issues early |

---

## Project Structure

```
Directory structure:
└── rahul2008d-rag-system-newpage/
    ├── README.md
    ├── Dockerfile
    ├── main.py
    ├── pyproject.toml
    ├── pytest.ini
    ├── ruff.toml
    ├── .dockerignore
    ├── .pre-commit-config.yaml
    ├── .pylintrc
    ├── .python-version
    └── src/
        ├── app.py
        └── rag/
            ├── __init__.py
            ├── chunking.py
            ├── config.py
            ├── loaders.py
            ├── store.py
            └── system.py

```


---

## ⚙️ How It Works (RAG Flow)

### 1. Ingestion
- Load documents using LangChain loaders
- Split text into overlapping chunks
- Generate embeddings using OpenAI
- Store vectors locally in FAISS

### 2. Retrieval
- Embed user query
- Perform Top-K similarity search
- Apply confidence threshold

### 3. Generation
- Build prompt using retrieved context
- Instruct model to answer **only from context**
- Cite chunk IDs inline

If no chunk meets the confidence threshold, the system responds with:

> *"I don't know based on the provided documents."*

---

## 🔐 Configuration

All configuration is environment-driven.

```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
TOP_K=5
MIN_SCORE=0.25
STORE_DIR=vector_store
```

## 🐳 Running Locally (Docker)

```
docker build -t chat-with-docs .
docker run -p 8501:8501 --env-file .env chat-with-docs
```

## Open in browser:

```
http://localhost:8501
```

---

## ☁️ Deployment (AWS ECS Fargate)

🚀 **[Open Live Application](https://YOUR_PUBLIC_URL)**

The application is deployed as a **serverless container** using **AWS ECS with the Fargate launch type**, requiring no EC2 or infrastructure management.

### Deployment Setup
- **Container Image**
  - Built using a multi-stage Dockerfile
  - Stored in **Amazon ECR**
  - Versioned image pushed and referenced by ECS task definition

- **ECS Configuration**
  - ECS Cluster with **Fargate** capacity provider
  - ECS Service running a single task (demo-friendly)
  - **0.5 vCPU / 1 GB RAM** per task
  - Public access enabled via **Application Load Balancer**
  - Health checks managed by ECS

- **Runtime Configuration**
  - Environment variables injected at runtime (e.g. `OPENAI_API_KEY`)
  - No secrets baked into the image
  - Stateless container design

- **Persistence & State**
  - Vector store backed by **FAISS** on the container filesystem
  - Re-ingestion required on task restart (intentional trade-off for simplicity)
  - Designed for easy upgrade to S3 / EFS if persistence is required

### Cost & Scaling
- On-demand billing (pay only while the task is running)
- Service can be **paused instantly** by setting:


## Cost Control

- Service can be paused by setting Desired tasks = 0
- No compute cost when stopped
- OpenAI billed strictly per token usage
---

## 🧪 Engineering Standards

- Ruff – linting and formatting
- Pylint – design-focused checks
- pre-commit hooks – enforced locally
- Clear module boundaries
- Minimal hidden magic

## 🔮 What I’d Add With More Time

- Persistent storage (S3 or EFS)
- Authentication and multi-tenant isolation
- Background ingestion jobs

## 📌 Summary

This project demonstrates how to build a clean, testable RAG system that can evolve into a larger platform without rewriting core components.
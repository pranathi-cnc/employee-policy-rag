# Employee Policy Question Answering

## Overview

A small RAG-based employee policy assistant that answers questions from three company policy documents.

The system uses **EmbeddingGemma** for embeddings, **ChromaDB** for retrieval, and **Gemma 4** for generating grounded answers.

## Policies

- `leave_policy.txt`
- `reimbursement_policy.txt`
- `working_hours_policy.txt`

## Tech Stack

- Python
- Ollama
- Gemma 4
- EmbeddingGemma
- ChromaDB
- Gradio

## RAG Workflow

```text
Policy Documents
      ↓
Chunking
      ↓
EmbeddingGemma
      ↓
ChromaDB
      ↓
User Question
      ↓
Retrieve Top 3 Chunks
      ↓
Gemma 4
      ↓
Answer + Source

Project Structure
employee-policy-qa/
│
├── data/
│   ├── leave_policy.txt
│   ├── reimbursement_policy.txt
│   ├── working_hours_policy.txt
│   └── questions.txt
│
├── src/
│   ├── app.py
│   └── evaluate.py
│
├── chroma_db/
├── outputs/
├── .env
├── requirements.txt
└── README.md
Setup

Create and activate the virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install ollama chromadb python-dotenv gradio

Pull the required Ollama models:

ollama pull gemma4:latest
ollama pull embeddinggemma
Environment Variables

Create .env:

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:latest
EMBEDDING_MODEL=embeddinggemma
Run the Application
python src\app.py

Open the Gradio URL shown in the terminal.

http://127.0.0.1:7860

Evaluation

The project contains 10 reference questions and 2 questions with missing information.

Run:

python src\evaluate.py

The evaluation checks:

Leave policy questions
Reimbursement policy questions
Working-hours questions
Missing-information questions
Retrieved source filenames
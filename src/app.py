import os
from pathlib import Path

import chromadb
import ollama
from dotenv import load_dotenv
import gradio as gr

# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma")

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

ollama_client = ollama.Client(host=OLLAMA_URL)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = chroma_client.get_or_create_collection(
    name="employee_policies"
)


# -----------------------------
# Create chunks
# -----------------------------

def create_chunks(text, chunk_size=300):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# Generate embedding
# -----------------------------

def get_embedding(text):
    response = ollama_client.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )

    return response["embedding"]


# -----------------------------
# Load policies into Chroma
# -----------------------------

def load_documents():

    # Avoid adding duplicate chunks every time
    if collection.count() > 0:
        print(f"ChromaDB already contains {collection.count()} chunks.")
        return

    chunk_id = 0

    for file_path in DATA_DIR.glob("*.txt"):

        text = file_path.read_text(encoding="utf-8")

        chunks = create_chunks(text)

        for chunk in chunks:

            embedding = get_embedding(chunk)

            collection.add(
                ids=[f"chunk_{chunk_id}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source": file_path.name
                    }
                ]
            )

            chunk_id += 1

    print(f"Loaded {chunk_id} chunks into ChromaDB.")


# -----------------------------
# Retrieve relevant chunks
# -----------------------------

def search_documents(question, top_k=3):

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved_chunks = []

    for document, metadata in zip(documents, metadatas):

        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata["source"]
            }
        )

    return retrieved_chunks


# -----------------------------
# Generate answer with Gemma
# -----------------------------

def answer_question(question):

    retrieved_chunks = search_documents(question)

    context = "\n\n".join(
        f"Source: {item['source']}\n"
        f"{item['text']}"
        for item in retrieved_chunks
    )

    prompt = f"""
You are an employee policy assistant.

Answer the question ONLY using the supplied policy text.

Question:
{question}

Supplied policy text:
{context}

Rules:
- Do not use outside knowledge.
- Do not guess.
- If the answer is not present in the supplied policy text, reply exactly:
  The supplied policies do not contain the relevant information.
- Return the source filename when information is available.
"""

    response = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt
    )

    answer = response["response"].strip()

    return answer, retrieved_chunks


# -----------------------------
# Main CLI
# -----------------------------
# -----------------------------
# Gradio UI
# -----------------------------

def ask_policy(question):

    if not question.strip():
        return "Please enter a question.", "", ""

    answer, retrieved_chunks = answer_question(question)

    sources = sorted(
        set(item["source"] for item in retrieved_chunks)
    )

    source_text = "\n".join(
        f"- {source}" for source in sources
    )

    retrieved_text = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in retrieved_chunks
    )

    return answer, source_text, retrieved_text


# Load policies when application starts
load_documents()


demo = gr.Interface(
    fn=ask_policy,
    inputs=gr.Textbox(
        label="Employee Policy Question",
        placeholder="Ask about leave, reimbursement, or working hours..."
    ),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources"),
        gr.Textbox(label="Retrieved Text"),
    ],
    title="Employee Policy Assistant",
    description="Ask questions based on the company employee policies.",
)


if __name__ == "__main__":
    demo.launch()
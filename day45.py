# Day 45 — RAG: Retrieval Augmented Generation
# Week 10 Day 1
#
# TOPIC 1: Claude has no knowledge of private documents — without RAG it either
# makes things up or tells you to ask HR. Useless for a real product.
#
# TOPIC 2: Build a knowledge base by loading docs, splitting into chunks, and
# scoring each chunk against the query by counting overlapping words. Top matches
# are retrieved — no embeddings or external APIs required.
#
# TOPIC 3: Pass retrieved chunks as context in the user message. The system prompt
# tells Claude to answer only from that context. Claude now gives exact, sourced
# answers instead of guesses.

import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ============================================================
# TOPIC 1: The problem — Claude doesn't know your documents
# ============================================================

def ask_without_rag(question):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

if __name__ == "__main__":
    questions = [
        "What is Acme Corp's daily meal allowance during business travel?",
        "How many days do I have to submit an expense report at Acme Corp?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask_without_rag(q)}")
        print()

# ============================================================
# TOPIC 2: Building the knowledge base — load, chunk, search
# ============================================================

def load_documents(folder="docs"):
    docs = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(f"{folder}/{filename}") as f:
                docs[filename] = f.read()
    return docs

def chunk_document(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_knowledge_base(folder="docs"):
    docs = load_documents(folder)
    knowledge_base = []
    for filename, text in docs.items():
        for chunk in chunk_document(text):
            knowledge_base.append({"source": filename, "text": chunk})
    return knowledge_base

def search(query, knowledge_base, top_k=2):
    query_words = set(query.lower().split())
    scored = []
    for chunk in knowledge_base:
        chunk_words = set(chunk["text"].lower().split())
        score = len(query_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]

if __name__ == "__main__":
    kb = build_knowledge_base()
    print(f"Knowledge base: {len(kb)} chunks loaded\n")

    query = "meal allowance business travel"
    results = search(query, kb)
    for r in results:
        print(f"Source: {r['source']}")
        print(f"Chunk: {r['text'][:200]}")
        print()

# ============================================================
# TOPIC 3: The RAG pipeline — retrieve, then generate
# ============================================================

def ask_with_rag(question, knowledge_base):
    chunks = search(question, knowledge_base)
    context = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in chunks])

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system="Answer questions using only the provided context. If the answer isn't in the context, say 'I don't have that information in the policy documents.'",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text

if __name__ == "__main__":
    kb = build_knowledge_base()
    print(f"Knowledge base: {len(kb)} chunks loaded\n")

    questions = [
        "What is Acme Corp's daily meal allowance during business travel?",
        "How many days do I have to submit an expense report at Acme Corp?",
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"Without RAG: {ask_without_rag(q)[:100]}...")
        print(f"With RAG:    {ask_with_rag(q, kb)}")
        print()

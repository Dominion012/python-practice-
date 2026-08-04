# Day 46 — Embeddings and Semantic Search
# Week 10 Day 2
#
# TOPIC 1: Embeddings are lists of 512 numbers that represent the meaning of text.
# Voyage AI's voyage-3-lite model converts any string into one of these vectors.
# Similar sentences produce similar vectors — even with completely different words.
#
# TOPIC 2: Cosine similarity compares two vectors and returns a score 0-1.
# Close to 1 = similar meaning. Close to 0 = unrelated. This is how semantic
# search knows "meal allowance" matches "reimbursed $75 per day".
#
# TOPIC 3: Semantic RAG embeds all document chunks upfront, then embeds the query
# at search time and picks the chunks with highest cosine similarity. Fixes the
# keyword search failure from Day 45 — meaning matches, not just words.

import os
import time 
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
import voyageai

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

# ============================================================
# TOPIC 1: What embeddings are — text as vectors of numbers
# ============================================================

sentences = [
    "Meals during business travel are reimbursed up to $75 per day",
    "What is the daily food allowance for work trips?",
    "The guest WiFi password is reset every 90 days",
]

result = vo.embed(sentences, model="voyage-3-lite")

for sentence, embedding in zip(sentences, result.embeddings):
    print(f"Text: {sentence[:50]}...")
    print(f"Vector length: {len(embedding)}")
    print(f"First 5 values: {[round(x, 4) for x in embedding[:5]]}")
    print()
# ============================================================
# TOPIC 2: Cosine similarity — measuring meaning distance
# ============================================================

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    embeddings = result.embeddings

    meal_doc   = embeddings[0]  # "Meals reimbursed $75 per day"
    meal_query = embeddings[1]  # "daily food allowance for work trips"
    wifi_doc   = embeddings[2]  # "WiFi password reset every 90 days"

    print("Similarity scores:")
    print(f"Meal doc vs meal query: {cosine_similarity(meal_doc, meal_query):.4f}")
    print(f"Meal doc vs WiFi doc:   {cosine_similarity(meal_doc, wifi_doc):.4f}")

# ============================================================
# TOPIC 3: Semantic RAG — embedding-based retrieval
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
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def build_semantic_kb(folder="docs"):
    docs = load_documents(folder)
    chunks = []
    for filename, text in docs.items():
        for chunk in chunk_document(text):
            chunks.append({"source": filename, "text": chunk})

    texts = [c["text"] for c in chunks]
    embeddings = vo.embed(texts, model="voyage-3-lite").embeddings
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]
    return chunks

def semantic_search(query, knowledge_base, top_k=2):
    
    query_embedding = vo.embed([query], model="voyage-3-lite").embeddings[0]
    scored = [(cosine_similarity(query_embedding, c["embedding"]), c) for c in knowledge_base]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]

def ask_with_semantic_rag(question, knowledge_base):
    chunks = semantic_search(question, knowledge_base)
    context = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in chunks])
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system="Answer questions using only the provided context. If the answer isn't in the context, say 'I don't have that information in the policy documents.'",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )
    return response.content[0].text

if __name__ == "__main__":
    # ... keep Topic 1 + 2 code ...

    print("\n--- Semantic RAG ---")
    kb = build_semantic_kb()
    print(f"Knowledge base: {len(kb)} chunks\n")
    

    questions = [
        "What is the meal allowance?",
        "How many days to submit an expense report?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask_with_semantic_rag(q, kb)}")
        print()

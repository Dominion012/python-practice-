# Day 47 — Conversational RAG
# Week 10 Day 3
#
# TOPIC 1: Single-turn RAG has no memory — each question is independent.
# Claude retrieves context and answers, but forgets everything next call.
# Works for simple lookups, fails for follow-up questions.
#
# TOPIC 2: ConversationalRAG adds self.messages persistence. Before each user
# message, relevant chunks are retrieved and injected as context. Claude sees
# both the retrieved docs AND the full conversation history every turn.
#
# TOPIC 3: Embeddings are expensive to recompute. save_kb() serialises the
# vector store to JSON; load_kb() restores it on the next run. First run builds
# and saves, every subsequent run loads from cache instantly.

import os
import json
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
import voyageai

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

# ============================================================
# TOPIC 1: The limitation — single-turn RAG forgets context
# ============================================================

def single_turn_rag(question, kb):
    query_emb = vo.embed([question], model="voyage-3-lite").embeddings[0]
    scored = [(np.dot(np.array(query_emb), np.array(c["embedding"])) /
               (np.linalg.norm(query_emb) * np.linalg.norm(c["embedding"])), c)
              for c in kb]
    scored.sort(key=lambda x: x[0], reverse=True)
    chunks = [c for _, c in scored[:2]]
    context = "\n\n".join([c["text"] for c in chunks])
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system="Answer using only the provided context.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    )
    return response.content[0].text

# ============================================================
# TOPIC 2: Conversational RAG — memory + semantic retrieval
# ============================================================

class ConversationalRAG:
    def __init__(self, kb):
        self.kb = kb
        self.messages = []

    def search(self, query, top_k=2):
        query_emb = vo.embed([query], model="voyage-3-lite").embeddings[0]
        scored = [(np.dot(np.array(query_emb), np.array(c["embedding"])) /
                   (np.linalg.norm(query_emb) * np.linalg.norm(c["embedding"])), c)
                  for c in self.kb]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def chat(self, question):
        chunks = self.search(question)
        context = "\n\n".join([c["text"] for c in chunks])
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        self.messages.append({"role": "user", "content": user_message})
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            system="Answer using only the provided context. You can reference earlier answers in this conversation.",
            messages=self.messages
        )
        answer = response.content[0].text
        self.messages.append({"role": "assistant", "content": answer})
        return answer

# ============================================================
# TOPIC 3: Caching the vector store — don't recompute every run
# ============================================================

def save_kb(kb, filepath="vector_store.json"):
    with open(filepath, "w") as f:
        json.dump(kb, f)

def load_kb(filepath="vector_store.json"):
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    from day46_exercise import build_kb, PRODUCT_DOCS

    kb = load_kb()
    if not kb:
        print("Building knowledge base...")
        kb = build_kb(PRODUCT_DOCS)
        save_kb(kb)
    else:
        print("Loaded from cache.")

    print("--- Conversational RAG ---")
    rag = ConversationalRAG(kb)

    questions = [
        "How much is the Pro plan?",
        "What about the Basic plan?",
        "Which of those two did you mention first?",
    ]
    for q in questions:
        print(f"You: {q}")
        print(f"Bot: {rag.chat(q)}")
        print()

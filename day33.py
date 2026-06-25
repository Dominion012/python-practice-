# Day 33 — Polishing the RAG System
# Week 7 Day 3

import os
import json
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
model = SentenceTransformer('all-MiniLM-L6-v2')


def load_documents(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename)) as f:
                documents[filename] = f.read()
    return documents


def chunk_by_paragraph(text, min_length=50):
    # merge short fragments (like headers) into the next paragraph
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged, buffer = [], ""
    for p in raw_paragraphs:
        buffer = f"{buffer} {p}".strip() if buffer else p
        if len(buffer) >= min_length:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


class DocumentStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.sources = []   # parallel to self.chunks
        self.embeddings = None

    def add(self, chunks, source):
        self.chunks.extend(chunks)
        self.sources.extend([source] * len(chunks))
        new_embeddings = self.embedding_model.encode(chunks)
        self.embeddings = new_embeddings if self.embeddings is None else np.vstack([self.embeddings, new_embeddings])

    def search(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query])[0]
        normalized_chunks = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = normalized_chunks @ normalized_query
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(similarities[i])}
            for i in top_indices
        ]


def search_with_threshold(store, query, top_k=3, min_score=0.3):
    return [r for r in store.search(query, top_k=top_k) if r["score"] >= min_score]


def build_rag_prompt_with_sources(query, results):
    context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
    return f"""Answer the question using ONLY the context below. If the context doesn't contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""


def rag_answer_with_citations(store, query, min_score=0.3, top_k=3):
    results = search_with_threshold(store, query, top_k=top_k, min_score=min_score)
    if not results:
        print(f"Q: {query}\nA: I don't have enough information to answer that.\n")
        return

    prompt = build_rag_prompt_with_sources(query, results)
    response = client.messages.create(
        model="claude-haiku-4-5", max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"Q: {query}")
    print(f"A: {response.content[0].text}")
    print(f"Sources used: {sorted(set(r['source'] for r in results))}\n")


class RAGChatbot:
    def __init__(self, store, client, min_score=0.3, top_k=3):
        self.store = store
        self.client = client
        self.min_score = min_score
        self.top_k = top_k
        self.history = []
        self.last_query = ""   # folded into retrieval so vague follow-ups still work

    def ask(self, query):
        retrieval_query = f"{self.last_query} {query}".strip()
        results = search_with_threshold(self.store, retrieval_query, top_k=self.top_k, min_score=self.min_score)

        if results:
            context = "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
            system_context = f"Answer using ONLY this context. If it doesn't contain the answer, say so:\n\n{context}"
        else:
            system_context = "No relevant context was found. Say you don't have enough information."

        self.history.append({"role": "user", "content": query})
        response = self.client.messages.create(
            model="claude-haiku-4-5", max_tokens=200, temperature=0,
            system=system_context, messages=self.history
        )
        answer = response.content[0].text
        self.history.append({"role": "assistant", "content": answer})
        self.last_query = query
        return answer


def save_store(store, prefix="vector_store"):
    np.save(f"{prefix}_embeddings.npy", store.embeddings)
    with open(f"{prefix}_metadata.json", "w") as f:
        json.dump({"chunks": store.chunks, "sources": store.sources}, f)


def load_store(embedding_model, prefix="vector_store"):
    loaded = DocumentStore(embedding_model)
    loaded.embeddings = np.load(f"{prefix}_embeddings.npy")
    with open(f"{prefix}_metadata.json") as f:
        metadata = json.load(f)
    loaded.chunks = metadata["chunks"]
    loaded.sources = metadata["sources"]
    return loaded


def get_or_build_store(embedding_model, docs_folder="docs", prefix="vector_store"):
    if os.path.exists(f"{prefix}_embeddings.npy"):
        print("Loading saved vector store...")
        return load_store(embedding_model, prefix)

    print("Building vector store from documents...")
    documents = load_documents(docs_folder)
    new_store = DocumentStore(embedding_model)
    for filename, content in documents.items():
        new_store.add(chunk_by_paragraph(content), source=filename)
    save_store(new_store, prefix)
    return new_store


if __name__ == "__main__":
    doc_store = get_or_build_store(model)
    chatbot = RAGChatbot(doc_store, client)

    print("=== Acme Corp Policy Assistant ===")
    while True:
        user_input = input("\nAsk a question (or 'quit'): ")
        if user_input.lower() == "quit":
            break
        print(f"\n{chatbot.ask(user_input)}")

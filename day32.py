# Day 32 — Retrieval-Augmented Generation (RAG)
# Week 7 Day 2

import os
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
model = SentenceTransformer('all-MiniLM-L6-v2')

# ============================================================
# TOPIC 1: What RAG Is and Why It Matters
# ============================================================
# LLMs have zero access to private/post-training-cutoff documents. Naively
# pasting whole documents into every prompt is expensive and dilutes
# self-attention (Day 30) across mostly-irrelevant tokens. RAG retrieves
# only the relevant pieces for a specific question, then injects just those
# into the prompt. Bad retrieval can be WORSE than no retrieval: a model
# handed irrelevant context tends to force a confident-but-wrong answer
# rather than admitting it doesn't know — Topic 7's escape hatch is the fix.

# ============================================================
# TOPIC 2: Embeddings for Semantic Search
# ============================================================
# A sentence embedding model maps a whole chunk of text to ONE fixed-size
# vector capturing its meaning — similar meaning -> similar vector, even
# with zero word overlap. Different from Day 30's token embeddings (used
# internally by the transformer) — these are for comparing whole texts.

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",      # similar MEANING, almost no shared words
    "The stock market crashed today.",  # unrelated meaning
]
embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")   # (3, 384)

# ============================================================
# TOPIC 3: Cosine Similarity
# ============================================================
# cos_sim(A, B) = (A . B) / (||A|| * ||B||) — same dot product as Day 30's
# Q @ K.T self-attention score, just normalized to a -1..1 range so it's
# comparable regardless of vector magnitude.

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Cat/mat <-> Feline/rug:      {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
print(f"Cat/mat <-> Stock market:    {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
print(f"Feline/rug <-> Stock market: {cosine_similarity(embeddings[1], embeddings[2]):.4f}")

# ============================================================
# TOPIC 4: Chunking Documents
# ============================================================
# Embedding a whole long document as one vector blurs together every topic
# it covers. Split into smaller chunks first so each one embeds to a
# precise, specific meaning. Fixed-size character chunking is simple but
# can cut mid-word/mid-sentence, degrading the embedding's quality — real
# systems chunk on sentence/paragraph boundaries instead.

def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap   # overlap prevents info loss at boundaries
    return chunks

# ============================================================
# TOPIC 5 & 6: Vector Store + Retrieval
# ============================================================
# A "vector database" at small scale is just embeddings + matching text,
# searchable via one vectorized matrix multiplication against ALL stored
# vectors at once (Day 26's "forward pass = matmul", applied to search).

class SimpleVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = None

    def add(self, chunks):
        self.chunks.extend(chunks)
        new_embeddings = self.embedding_model.encode(chunks)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query, top_k=2):
        query_embedding = self.embedding_model.encode([query])[0]
        normalized_chunks = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = normalized_chunks @ normalized_query    # one matmul vs ALL chunks at once
        top_indices = np.argsort(similarities)[::-1][:top_k]    # highest similarity first
        return [(self.chunks[i], similarities[i]) for i in top_indices]

# ============================================================
# TOPIC 7: Augmentation — Building the Grounded Prompt
# ============================================================
# Inject retrieved chunks into a template that instructs the model to
# answer using ONLY that context, with an explicit escape hatch for when
# the context doesn't contain the answer — prevents confident hallucination.

def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return f"""Answer the question using ONLY the context below. If the context doesn't contain enough information to answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

# ============================================================
# TOPIC 8: Capstone — Full RAG System
# ============================================================

knowledge_base = [
    "Acme Corp's guest WiFi password is reset every 90 days. The current password can be found in the IT portal under Network Settings.",
    "Employees are entitled to 15 days of paid vacation per year, accruing monthly. Unused vacation days roll over up to a maximum of 5 days.",
    "Expense reports must be submitted within 30 days of the purchase date. Reports over $500 require manager approval before reimbursement.",
    "The office is open Monday through Friday, 8am to 6pm. Badge access is required outside of core hours of 9am to 5pm.",
    "New hires receive a laptop, monitor, and standard software license within their first week. IT setup requests go through the onboarding portal.",
    "Remote work is permitted up to 3 days per week, subject to manager approval. Fully remote arrangements require VP-level sign-off.",
]

policy_store = SimpleVectorStore(model)
policy_store.add(knowledge_base)

def rag_answer(query, top_k=2):
    results = policy_store.search(query, top_k=top_k)
    retrieved_chunks = [chunk for chunk, _ in results]
    prompt = build_rag_prompt(query, retrieved_chunks)

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        temperature=0,        # Day 31 Topic 7 — consistent, factual extraction
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"Q: {query}")
    print("Retrieved chunks:")
    for chunk, score in results:
        print(f"  [{score:.3f}] {chunk}")
    print(f"A: {response.content[0].text}\n")

if __name__ == "__main__":
    rag_answer("What is the WiFi password reset interval?")
    rag_answer("How many vacation days do I get?")
    rag_answer("Can I work from home?")
    rag_answer("What is the CEO's favorite food?")   # deliberately not in the knowledge base

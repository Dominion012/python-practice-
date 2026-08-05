# Day 48 — Week 10 Capstone: Company Knowledge Assistant
# Week 10 Day 4
#
# TOPIC 1: KnowledgeAssistant combines Day 43's strong system prompt with
# Day 47's conversational RAG class. The persona constrains Claude to stay on
# topic; semantic search grounds answers in real policy documents.
#
# TOPIC 2: chat_stream() replaces chat() for live token output. Uses
# client.messages.stream() to print text as it arrives, then get_final_message()
# to collect the full answer for self.messages persistence.
#
# TOPIC 3: Eval suite creates a fresh assistant per test case (no memory bleed),
# asks known questions, and scores answers with LLM-as-judge. Proves the full
# pipeline works reliably before handing it to a client.

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
# TOPIC 1: RAG agent with persona
# ============================================================

SYSTEM_PROMPT = """You are Alex, Acme Corp's internal knowledge assistant.

Your job:
- Answer questions about Acme Corp policies using only the provided context
- Be concise — 2-3 sentences max
- If the answer isn't in the context, say "I don't have that information in the policy documents"

Never:
- Make up policy details
- Give legal or financial advice
- Answer questions unrelated to Acme Corp
"""

def load_documents(folder="docs"):
    docs = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(f"{folder}/{filename}") as f:
                docs[filename] = f.read()
    return docs

def chunk_document(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_kb(folder="docs"):
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

def save_kb(kb, filepath="kb_cache.json"):
    with open(filepath, "w") as f:
        json.dump(kb, f)

def load_kb(filepath="kb_cache.json"):
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class KnowledgeAssistant:
    def __init__(self, kb):
        self.kb = kb
        self.messages = []

    def search(self, query, top_k=2):
        query_emb = vo.embed([query], model="voyage-3-lite").embeddings[0]
        scored = [(cosine_similarity(query_emb, c["embedding"]), c) for c in self.kb]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def chat(self, question):
        chunks = self.search(question)
        context = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in chunks])
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        self.messages.append({"role": "user", "content": user_message})
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=self.messages
        )
        answer = response.content[0].text
        self.messages.append({"role": "assistant", "content": answer})
        return answer
    def chat_stream(self, question):
        chunks = self.search(question)
        context = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in chunks])
        user_message = f"Context:\n{context}\n\nQuestion: {question}"
        self.messages.append({"role": "user", "content": user_message})

        print("Alex: ", end="", flush=True)
        with client.messages.stream(
            model="claude-haiku-4-5",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=self.messages
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            final = stream.get_final_message()
        print()

        answer = final.content[0].text
        self.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    kb = load_kb()
    if not kb:
        print("Building knowledge base...")
        kb = build_kb()
        save_kb(kb)
    else:
        print("Loaded from cache.")

    assistant = KnowledgeAssistant(kb)
    print("Alex: Hi! I'm Alex, Acme Corp's knowledge assistant.\n")

    questions = [
        "What is the meal allowance for business travel?",
        "How many days do I have to submit an expense report?",
        "Can I fly business class?",
    ]
    for q in questions:
        print(f"You: {q}")
        assistant.chat_stream(q)
        print()

# ============================================================
# TOPIC 3: Eval suite — testing the full RAG pipeline
# ============================================================

EVAL_CASES = [
    {
        "question": "What is the hotel cap per night?",
        "ideal": "$250 per night in most cities"
    },
    {
        "question": "When is business class allowed on flights?",
        "ideal": "Flights longer than 6 hours with manager pre-approval"
    },
    {
        "question": "What approval is needed for expense reports over $500?",
        "ideal": "Manager approval before reimbursement is processed"
    },
]

def judge(question, answer, ideal):
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        system="You are a strict grader. Score the answer 1-5 based on correctness. Reply with just a number.",
        messages=[{"role": "user", "content": f"Question: {question}\nIdeal: {ideal}\nAnswer: {answer}\nScore:"}]
    )
    return int(response.content[0].text.strip()[0])

def run_evals(kb):
    total = 0
    for tc in EVAL_CASES:
        assistant = KnowledgeAssistant(kb)
        answer = assistant.chat(tc["question"])
        score = judge(tc["question"], answer, tc["ideal"])
        total += score
        print(f"[{score}/5] {tc['question']}")
        print(f"       {answer[:80]}...")
    print(f"\nAverage: {total / len(EVAL_CASES):.1f}/5")

if __name__ == "__main__":
    kb = load_kb()
    if not kb:
        print("Building knowledge base...")
        kb = build_kb()
        save_kb(kb)
    else:
        print("Loaded from cache.")

    assistant = KnowledgeAssistant(kb)
    print("Alex: Hi! I'm Alex, Acme Corp's knowledge assistant.\n")

    questions = [
        "What is the meal allowance for business travel?",
        "How many days do I have to submit an expense report?",
        "Can I fly business class?",
    ]
    for q in questions:
        print(f"You: {q}")
        assistant.chat_stream(q)

        print()
    run_evals(kb)

    


# Day 49 — Deploying Your Agent as a Web API
# Week 11 Day 1

import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from day48 import KnowledgeAssistant, load_kb, build_kb, save_kb

app = FastAPI()

# Load KB once at startup
kb = load_kb()
if not kb:
    print("Building knowledge base...")
    kb = build_kb()
    save_kb(kb)

assistant = KnowledgeAssistant(kb)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"message": "Agent API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    answer = assistant.chat(request.question)
    return ChatResponse(answer=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

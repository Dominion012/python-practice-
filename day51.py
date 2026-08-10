# Day 51 — Streaming API Responses
# Week 11 Day 3

from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from anthropic import Anthropic
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
app = FastAPI()

class ChatRequest(BaseModel):
    question: str

def stream_claude(question):
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": question}]
    ) as stream:
        for text in stream.text_stream:
            yield text

@app.post("/chat-stream")
def chat_stream(request: ChatRequest):
    return StreamingResponse(stream_claude(request.question), media_type="text/plain")

from day48 import KnowledgeAssistant, load_kb, build_kb, save_kb

API_KEY = os.environ["API_KEY"]

kb = load_kb()
if not kb:
    kb = build_kb()
    save_kb(kb)

assistant = KnowledgeAssistant(kb)

def verify_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def stream_rag(question):
    chunks = assistant.search(question)
    context = "\n\n".join([c["text"] for c in chunks])
    user_message = f"Context:\n{context}\n\nQuestion: {question}"
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for text in stream.text_stream:
            yield text

@app.post("/chat-rag")
def chat_rag(request: ChatRequest, _=Depends(verify_key)):
    return StreamingResponse(stream_rag(request.question), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

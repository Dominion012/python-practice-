# Day 53 — Week 11 Capstone: Production AI API
# Week 11 Day 5

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from anthropic import Anthropic
import uvicorn
from day48 import KnowledgeAssistant, load_kb, build_kb, save_kb

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
API_KEY = os.environ["API_KEY"]

app = FastAPI(title="Acme Knowledge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

kb = load_kb()
if not kb:
    print("Building knowledge base...")
    kb = build_kb()
    save_kb(kb)

assistant = KnowledgeAssistant(kb)

class ChatRequest(BaseModel):
    question: str

def verify_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", dependencies=[Depends(verify_key)])
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = assistant.chat(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def stream_response(question):
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

@app.post("/chat-stream", dependencies=[Depends(verify_key)])
def chat_stream(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return StreamingResponse(stream_response(request.question), media_type="text/plain")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)




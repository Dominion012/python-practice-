# Day 50 — API Authentication
# Week 11 Day 2

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
API_KEY = os.environ["API_KEY"]

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"answer": f"You asked: {request.question}"}

def verify_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health", dependencies=[Depends(verify_key)])
def health():
    return {"status": "ok"}

@app.post("/chat-secure")
def chat_secure(request: ChatRequest, _=Depends(verify_key)):
    return {"answer": f"Secured: {request.question}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

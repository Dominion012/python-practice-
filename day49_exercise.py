from fastapi import FastAPI
import os
from pydantic import BaseModel
from day48_exercise import ITHelpDesk, load, save, build_kb
import uvicorn


app = FastAPI()
kb = load(file="save_kb.json")
if not kb:
    print("building knowledge base ....")
    kb = build_kb()
    save(kb)

build = ITHelpDesk(kb)
@app.get("/")
def root():
    return {"message":"Agent is running"}

@app.get("/health")
def health():
    return {"status": "okay"}

class ChatRequest(BaseModel):
    question:str

class ChatResponse(BaseModel):
    reply:str

@app.post("/chat")
def agent(request:ChatRequest):
    answer = build.chat(request.question)
    return ChatResponse(reply = answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
from day48_exercise import ITHelpDesk, load, save , build_kb
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from anthropic import Anthropic
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
API_KEY = os.environ["API_KEY"]
kb = load(file="save_kb.json")
if not kb:
    
    print("Building knowledge Base........")
    kb = build_kb()
    save(kb)

bot = ITHelpDesk(kb)

class ChatRequest(BaseModel):
    question: str


def verify_key(x_api_key:str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code= 401, detail= "Invalid Api Key")
    
def stream_helpdesk(question):
    chunk = bot.search(question)
    context = "\n\n".join(c["text"] for c in chunk)
    user_message = f"Context:\n{context}\nQuestion:\n{question}"
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role":"user", "content":user_message}]
    ) as stream:
        for text in stream.text_stream:
            yield text


@app.post("/chat")
def chat_stream(query:ChatRequest, _=Depends(verify_key)):
    return StreamingResponse(stream_helpdesk(query.question), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
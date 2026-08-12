import os 
from fastapi import FastAPI, Depends,HTTPException, Header
from day48_exercise import ITHelpDesk, load, save, build_kb
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from anthropic import Anthropic
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()

app.add_middleware  (
    CORSMiddleware,
    allow_origins = ["*"],
    allow_headers = ["*"],
    allow_methods = ["*"]
)
API_KEY = os.environ["API_KEY"]


kb = load(file="save_kb.json")
if not kb:
    print("building knowledge base......")
    kb = build_kb()
    save(kb)

bot = ITHelpDesk(kb)

class ChatResponse(BaseModel):
    question:str
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def stream_helpdesk(question):
    chunks = bot.search(question)
    context = "\n\n".join(c["text"] for c in chunks)
    user_message = f"Context:\n{context}\nQuestion:\n{question}"
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for text in stream.text_stream:
            yield text

def verify(x_api_key:str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code= 401, detail= "Invalid Api Key")
    

@app.get("/health", dependencies=[Depends(verify)])
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(request:ChatResponse, _=Depends(verify)):
    if not request.question:
        raise HTTPException(status_code= 400, detail= "Question placeholder cannot be empty")
    try: 
        return {"answer" : bot.chat(request.question)}
    
    except Exception as e:
        raise HTTPException(status_code= 500, detail=str(e))
    

@app.post("/chat_stream")
def chat_stream(request:ChatResponse, _=Depends(verify)):
    if not request.question:
        raise HTTPException(status_code= 400, detail= "Question placeholder cannot be empty")
    try: 
        return  StreamingResponse(stream_helpdesk(request.question), media_type= "text/plain")
    
    except Exception as e:
        raise HTTPException(status_code= 500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host= "0.0.0.0", port = 8004)

from dotenv import load_dotenv
from day48_exercise import ITHelpDesk, load, save , build_kb
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
import os

load_dotenv()
app = FastAPI()
API_KEY = os.environ["API_KEY"]


kb = load(file = "save_kb.json")
if not kb:
    print("Building knowledge base......")
    kb = build_kb()
    save(kb)

bot = ITHelpDesk(kb)
class ChatRequest(BaseModel):
    question:str

class ChatResponse(BaseModel):
    answer:str


def verify(x_api_key:str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail= "invalid api key")
    
@app.get("/health", dependencies=[Depends(verify)])
def health():
    return {"status": "ok"}
@app.post("/chat")
def chat(request:ChatRequest, _=Depends(verify)):
    answer = bot.chat(request.question)
    return ChatResponse(answer=answer)

if __name__ =="__main__":
    uvicorn.run(app,host="0.0.0.0", port=8001)

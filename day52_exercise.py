from dotenv import load_dotenv
from day48_exercise import ITHelpDesk, load, save , build_kb
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os


app = FastAPI()
API_KEY = os.environ["API_KEY"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"])

kb = load(file = "save_kb.json")
if not kb:
    print("Building knowedge base......")
    kb = build_kb()
    save(kb)

bot = ITHelpDesk(kb)

class ChatRequest(BaseModel):
    question:str

def verify(x_api_key:str=Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail= "Invalid Api Key")
    

@app.post("/chat")
def chat(request:ChatRequest, _=Depends(verify)):
    r = request.question
    if not r :
        raise HTTPException(status_code= 400, detail= "Question cannot be empty")
    try:
        return {"answer" : bot.chat(request.question)}
    except Exception as e :
        raise HTTPException(status_code= 500, detail= f"Something went wrong: {str(e)}")
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0", port= 8003)

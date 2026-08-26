# Day 61 — Full AI Project: Auth-Protected RAG API

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Session
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
import datetime
import uvicorn
import os

app = FastAPI()
db_user = os.environ.get("DB_USER", os.environ.get("USER", "postgres"))
db_host = os.environ.get("DB_HOST", "localhost")
db_password = os.environ.get("DB_PASSWORD", "")
db_name = os.environ.get("DB_NAME", "mydb")
db_url = f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}" if db_password else f"postgresql://{db_user}@{db_host}/{db_name}"
engine = create_engine(db_url)
SECRET_KEY = "supersecretkey_atleast32charslong!!"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    user_email = Column(String)
    question = Column(Text)
    answer = Column(Text)

Base.metadata.create_all(engine)

class RegisterRequest(BaseModel):
    email: str
    password: str

def create_token(email: str):
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/register")
def register(request: RegisterRequest):
    with Session(engine) as session:
        existing = session.query(User).filter(User.email == request.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed = pwd_context.hash(request.password)
        user = User(email=request.email, hashed_password=hashed)
        session.add(user)
        session.commit()
        return {"message": f"User {request.email} registered"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        user = session.query(User).filter(User.email == form_data.username).first()
        if not user or not pwd_context.verify(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token(user.email)
        return {"access_token": token, "token_type": "bearer"}
import anthropic
from dotenv import load_dotenv
from day48 import KnowledgeAssistant, load_kb, build_kb, save_kb

load_dotenv()
client = anthropic.Anthropic()

kb = load_kb()
if not kb:
    kb = build_kb()
    save_kb(kb)
assistant = KnowledgeAssistant(kb)

class ChatRequest(BaseModel):
    question: str

def save_history(email: str, question: str, answer: str):
    with Session(engine) as session:
        entry = ChatHistory(user_email=email, question=question, answer=answer)
        session.add(entry)
        session.commit()

@app.post("/chat")
def chat(request: ChatRequest, payload: dict = Depends(get_current_user), background_tasks: BackgroundTasks = None):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = assistant.chat(request.question)
    background_tasks.add_task(save_history, payload["email"], request.question, answer)
    return {"answer": answer}

@app.get("/history")
def get_history(payload: dict = Depends(get_current_user)):
    with Session(engine) as session:
        entries = session.query(ChatHistory).filter(ChatHistory.user_email == payload["email"]).all()
        return [{"question": e.question, "answer": e.answer} for e in entries]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8018)

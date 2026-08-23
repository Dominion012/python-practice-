from fastapi import FastAPI, HTTPException, Header,Depends,BackgroundTasks
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import DeclarativeBase, Session
import jwt
import os 
from sqlalchemy import create_engine, Column, String, Integer,Boolean
import uvicorn
import datetime
app = FastAPI()
username = os.environ.get("USER", "postgres")
engine = create_engine(f"postgresql://{username}@localhost/mydb")
pwd_context = CryptContext(schemes=["bcrypt"],deprecated = "auto" )
outh2_scheme = OAuth2PasswordBearer(tokenUrl="login")
SECRET_KEY = "blababoshka_0124577_just _23notingto3rite"
class Base(DeclarativeBase):
    pass 

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key= True)
    email = Column(String, unique=True)
    hashed_password = Column(String)
    is_admin = Column(Boolean, default=False)


class Note(Base):
    __tablename__ = "note"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(String)
    owner_email = Column(String)

Base.metadata.create_all(engine)

class Reqrequest(BaseModel):
    email:str
    password:str
    is_admin: bool = False

class Noterequest(BaseModel):
    title:str
    content:str
    

def create_token(email:str):
    payload = {
        "email": email,
        "exp" : datetime.datetime.utcnow()+ datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

@app.post("/register")
def register_user(request:Reqrequest):
    with Session(engine) as session:
        user = session.query(User).filter(User.email == request.email).first()
        if user:
            raise HTTPException(status_code= 401, detail= "User already exists")
        hashed = pwd_context.hash(request.password)
        user = User(email= request.email, hashed_password = hashed, is_admin = request.is_admin)
        session.add(user)
        session.commit()
        return {"message": f"User with email {request.email} created"}
    

@app.post("/login")
def login(request:OAuth2PasswordRequestForm=Depends()):
    with Session(engine) as session:
        user= session.query(User).filter(User.email== request.username).first()
        if not user or not pwd_context.verify(request.password, user.hashed_password):
            raise HTTPException(status_code= 401, detail= "Invalid credentials")
        token = create_token(user.email)
        return {"access_token": token, "token_type" : "Bearer"}
    
def get_current_user(token:str= Depends(outh2_scheme)):
  
    try:
       payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
       return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code= 401, detail= "Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code= 401, detail= "Invalid token")
  
    
@app.post("/notes")
def add_notes (note: Noterequest,payload:dict = Depends(get_current_user), background_tasks:BackgroundTasks = None):
    email = payload["email"]
    with Session(engine) as session:
        new_note = Note(title = note.title, content=note.content, owner_email = email)
        background_tasks.add_task(log_notes, f"Note created: {note.title} by {email}")
        session.add(new_note)
        session.commit()
    return {"message": f"Note '{note.title} created"}

def log_notes(message: str):
    with open ("notes_log.txt", "a") as f:
        f.write(message +"\n")


@app.get("/notes")
def get_notes(payload:dict = Depends(get_current_user)):
    email = payload["email"]
    with Session(engine) as session:
        user = session.query(Note).filter(Note.owner_email == email).all()
        return [{"title": n.title, "content": n.content } for n in user ]
    
@app.get("/notes/all")
def get_all_notes(payload: dict = Depends(get_current_user)):
    email = payload["email"]

    with Session(engine) as session:
        user = session.query(User).filter(User.email == email).first()

        if not user or not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")

        notes = session.query(Note).all()

        return notes
    
@app.delete("/notes/{note_id}")
def delet_notes(note_id: int, payload: dict= Depends(get_current_user)):
    email = payload["email"]
    with Session(engine) as session:
        delete = session.query(Note).filter(Note.id == note_id, Note.owner_email == email).first()
        if not delete:
            raise HTTPException(status_code= 404, detail= "Note not found") 
        session.delete(delete)
        session.commit()

        return{"message": "Note deleted"}


if __name__ == "__main__":
    uvicorn.run(app, host= "0.0.0.0", port=8017)
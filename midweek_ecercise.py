from fastapi import FastAPI, Depends, BackgroundTasks,HTTPException,Header
from fastapi.security import OAuth2PasswordRequestForm,OAuth2PasswordBearer
from sqlalchemy import Column, String, Integer,Text, create_engine
from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase, Session
from passlib.context import CryptContext
import uvicorn
import os
import datetime
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request



app = FastAPI()
username = os.environ.get("USER", "postgres")
engine = create_engine(f"postgresql://{username}@localhost/mydb")
pwd_context = CryptContext(schemes=["bcrypt"],deprecated = "auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String)
    hashed_password = Column(String)

class Task (Base):
    __tablename__ = "task"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    status = Column(String, default="pending")
    user_email = Column(String)


Base.metadata.create_all(engine)

def log_task(task):
    with open("task_log.txt", "a") as f:
        f.write(task)

class RegisTar(BaseModel):
    email:str
    password:str

class TaskS(BaseModel):
    title:str
   
@app.post("/register")
def register(request:RegisTar):
    with Session(engine) as session:
        exist = session.query(User).filter(User.email == request.email).first()
        if exist:
            raise HTTPException(status_code= 401, detail= "User already exists")
        pwd = pwd_context.hash(request.password)
        new = User(email = request.email, hashed_password = pwd)
        session.add(new)
        session.commit()

        
        return {"message" : f"User created {request.email}"}
SECRET_KEY = "jnusbuvhew7huqe__-ieed68quinjdanu90"
def create_token(email:str):
    payload = {
        "email" : email,
        "exp" : datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def get_current_user(token:str=Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException (status_code= 401, detail= "Invalid token")
    except jwt.InvalidTokenError:
        raise HTTPException (status_code= 401, detail= "Token has expired")
    

@app.post("/login")
def login(form_data:OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        auth = session.query(User).filter(User.email == form_data.username).first()
        if not auth or not pwd_context.verify(form_data.password, auth.hashed_password):
            raise HTTPException(status_code= 401, detail= 'Invalid Credentials')
        token = create_token(auth.email)
        return {"access_token" : token, "token_type": "Bearer"}
    

    
@app.post("/tasks")
@limiter.limit("5/minute")
def tasks(request: Request, body: TaskS, payload: dict = Depends(get_current_user), background_tasks: BackgroundTasks = None):

    with Session(engine) as session:
        new_task = Task(title = body.title, user_email = payload["email"])
        background_tasks.add_task(log_task, f"Task created: {body.title} by {payload['email']}\n")

        session.add(new_task)
        session.commit()
        return {"message":f"{new_task.title} added"}
    
@app.get("/tasks")
def get_tasks(payload:dict= Depends(get_current_user)):
    with Session(engine) as session:
        task = session.query(Task).filter(Task.user_email == payload["email"]).all()
        if not task:
            raise HTTPException(status_code= 404, detail= "No tasks found")
        return [{"id": t.id, "title": t.title, "status": t.status} for t in task]


@app.put("/tasks/{task_id}")
def mod_tasks(task_id:int,payload:dict = Depends(get_current_user)):
    with Session(engine) as session:
        task = session.query(Task).filter(Task.user_email == payload["email"] ,Task.id == task_id).first()
        if not task:
            raise HTTPException (status_code= 404, detail= "Task not found")
        
        task.status = "done"
        session.commit()
        return {"message" : f"task{task_id} marked as done"}
    

@app.delete("/tasks/{task_id}")

def delete_tasks(task_id:int,payload:dict = Depends(get_current_user)):
    with Session(engine) as session:
        task = session.query(Task).filter(Task.user_email == payload["email"] ,Task.id == task_id).first()
        if not task:
            raise HTTPException (status_code= 404, detail= "Task not found")
        
        session.delete(task)
        session.commit()

        return {"message" : f"task{task_id} deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host= "0.0.0.0", port= 8020)
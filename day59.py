from fastapi import FastAPI, HTTPException, Header
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, Session
from passlib.context import CryptContext
import uvicorn
import os

app = FastAPI()
username = os.environ.get("USER", "postgres")
engine = create_engine(f"postgresql://{username}@localhost/mydb")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)

Base.metadata.create_all(engine)

# test hashing
hashed = pwd_context.hash("password123")
print("Hashed:", hashed)
print("Verified:", pwd_context.verify("password123", hashed))
print("Wrong password:", pwd_context.verify("wrongpass", hashed))

from pydantic import BaseModel

class RegisterRequest(BaseModel):
    email: str
    password: str

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

import jwt
import datetime

class LoginRequest(BaseModel):
    email: str
    password: str

SECRET_KEY = "supersecretkey_atleast32charslong!!"

def create_token(email: str):
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

@app.post("/login")
def login(request: LoginRequest):
    with Session(engine) as session:
        user = session.query(User).filter(User.email == request.email).first()
        if not user or not pwd_context.verify(request.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token(user.email)
        return {"access_token": token}

from fastapi import Depends

def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/me")
def get_me(payload: dict = Depends(verify_token)):
    return {"email": payload["email"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8013)



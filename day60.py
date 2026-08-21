from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.orm import DeclarativeBase, Session
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
import datetime
import uvicorn
import os

app = FastAPI()
username = os.environ.get("USER", "postgres")
engine = create_engine(f"postgresql://{username}@localhost/mydb")
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
    is_admin = Column(Boolean, default=False)

Base.metadata.create_all(engine)

class RegisterRequest(BaseModel):
    email: str
    password: str
    is_admin: bool = False

@app.post("/register")
def register(request: RegisterRequest):
    with Session(engine) as session:
        existing = session.query(User).filter(User.email == request.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed = pwd_context.hash(request.password)
        user = User(email=request.email, hashed_password=hashed, is_admin=request.is_admin)
        session.add(user)
        session.commit()
        return {"message": f"User {request.email} registered"}

def create_token(email: str):
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        user = session.query(User).filter(User.email == form_data.username).first()
        if not user or not pwd_context.verify(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token(user.email)
        return {"access_token": token, "token_type": "bearer"}


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/me")
def get_me(payload: dict = Depends(get_current_user)):
    return {"email": payload["email"]}

@app.get("/admin")
def admin_only(payload: dict = Depends(get_current_user)):
    with Session(engine) as session:
        user = session.query(User).filter(User.email == payload["email"]).first()
        if not user.is_admin:
            raise HTTPException(status_code=403, detail="Admin access only")
        return {"message": f"Welcome admin {user.email}!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8016)

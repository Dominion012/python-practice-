# Day 66 — Testing with pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "Hello, world!"}

@app.get("/add")
def add(a: int, b: int):
    return {"result": a + b}

# TOPIC 3: Testing with a database
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.pool import StaticPool

class Base(DeclarativeBase):
    pass

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True)
    name = Column(String)

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
Base.metadata.create_all(test_engine)

@app.post("/items")
def create_item(name: str):
    with Session(test_engine) as session:
        item = Item(name=name)
        session.add(item)
        session.commit()
        return {"id": item.id, "name": item.name}

@app.get("/items")
def list_items():
    with Session(test_engine) as session:
        items = session.query(Item).all()
        return [{"id": i.id, "name": i.name} for i in items]

# TOPIC 2: Testing endpoints with JWT auth
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt
import datetime

SECRET_KEY = "supersecretkey_atleast32charslong!!"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/protected")
def protected(payload: dict = Depends(get_current_user)):
    return {"email": payload["email"]}

def make_token(email: str):
    payload = {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# TOPIC 1: Basic tests
client = TestClient(app)

def test_hello():
    response = client.get("/hello")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, world!"}

def test_add():
    response = client.get("/add?a=3&b=4")
    assert response.status_code == 200
    assert response.json()["result"] == 7

def test_protected_with_valid_token():
    token = make_token("domi@test.com")
    response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "domi@test.com"

def test_protected_without_token():
    response = client.get("/protected")
    assert response.status_code == 401

def test_create_item():
    response = client.post("/items?name=controller")
    assert response.status_code == 200
    assert response.json()["name"] == "controller"

def test_list_items():
    client.post("/items?name=keyboard")
    response = client.get("/items")
    assert response.status_code == 200
    names = [i["name"] for i in response.json()]
    assert "keyboard" in names

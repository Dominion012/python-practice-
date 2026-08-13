# Day 54 — Testing Your API
# Week 12 Day 1

# TOPIC 1 & 2: pytest basics
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0

from fastapi.testclient import TestClient
from day53 import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_chat_no_key():
    response = client.post("/chat", json={"question": "test"})
    assert response.status_code == 401

def test_chat_with_key():
    response = client.post(
        "/chat",
        json={"question": "What is the meal allowance?"},
        headers={"x-api-key": "mysecretkey123"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()

from day53_exercise import app
from fastapi.testclient import TestClient


client = TestClient(app)

def test_health():
    response = client.get("/health", headers={"x-api-key": "mysecretkey123"})
    assert response.status_code == 200

def test_chat_no_key():
    response = client.post("/chat", json={"question": "test"})
    assert response.status_code == 401

def test_empty_question():
    response = client.post("/chat", json={"question": ""}, 
                           headers={"x-api-key": "mysecretkey123"})
    assert response.status_code == 400


def test_chat_with_key():
    response = client.post("/chat", json={"question": "where do i get a new badge"}, 
                           headers= {"x-api-key": "mysecretkey123"})
    assert response.status_code == 200
    assert "answer" in response.json()
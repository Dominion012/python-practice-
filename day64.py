# Day 64 — Caching with Redis

from fastapi import FastAPI
import redis
import time
import uvicorn

app = FastAPI()
cache = redis.Redis(host="localhost", port=6379, decode_responses=True)

# TOPIC 1: Cache a slow operation
def slow_operation(question: str):
    time.sleep(2)  # simulates a slow AI call or DB query
    return f"Answer to: {question}"

@app.get("/ask")
def ask(question: str):
    cached = cache.get(question)
    if cached:
        return {"answer": cached, "source": "cache"}
    
    answer = slow_operation(question)
    cache.set(question, answer, ex=60)  # cache for 60 seconds
    return {"answer": answer, "source": "fresh"}

# TOPIC 2: Cache expiry and manual invalidation
@app.get("/data")
def get_data(key: str):
    cached = cache.get(key)
    if cached:
        return {"data": cached, "source": "cache"}
    
    result = f"Fresh data for {key}"
    cache.set(key, result, ex=30)  # expires in 30 seconds
    return {"data": result, "source": "fresh"}

@app.delete("/cache/{key}")
def clear_cache(key: str):
    deleted = cache.delete(key)
    if deleted:
        return {"message": f"Cache cleared for {key}"}
    return {"message": "Key not found in cache"}

# TOPIC 3: Cache AI responses
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

@app.get("/ai")
def ai_answer(question: str):
    cached = cache.get(question)
    if cached:
        return {"answer": cached, "source": "cache"}
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": question}]
    )
    answer = response.content[0].text
    cache.set(question, answer, ex=300)  # cache for 5 minutes
    return {"answer": answer, "source": "fresh"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8022)

from fastapi import FastAPI
import uvicorn
import time
import redis


app = FastAPI()
cache = redis.Redis(host="localhost", port=6379, decode_responses=True)

from dotenv import load_dotenv
load_dotenv()
import anthropic
client = anthropic.Anthropic()
def slowresponse(topic:str):
    time.sleep(1)
    return f" News topic {topic}"

@app.get("/news/{topic}")
def news_topic(topic: str):
    cached = cache.get(topic)
    if cached:
        return {"headlines": cached, "source": "cached"}
    
    result = slowresponse(topic)
    cache.set(topic, result, ex=120)  # 2 minutes
    return {"headlines": result, "source": "fresh"}

@app.delete("/news/cache/{topic}")
def clear_news(topic:str):
    deleted = cache.delete(topic)
    if deleted:
        return {"message": f"Cache cleared for {topic}"}
    return{"message": "Topic not found in cache"}

@app.get("/news/ai/{topic}")
def ai_news(topic:str):
    cached = cache.get(topic)
    if cached: 
       return {"summary": cached, "source": "cached"}
    response = client.messages.create(model="claude-haiku-4-5-20251001",
                                      max_tokens= 300,
                                      messages=[{"role":"user", "content": topic}])
    reply =  response.content[0].text
    cache.set(topic, reply, ex=300)
    return{"summary":reply, "source": "fresh"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8023)
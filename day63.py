# Day 63 — Rate Limiting with SlowAPI

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# TOPIC 1: Limit an endpoint to 5 requests per minute
@app.get("/hello")
@limiter.limit("5/minute")
def hello(request: Request):
    return {"message": "Hello!"}

# TOPIC 2: Different limits per endpoint
@app.get("/free")
@limiter.limit("10/minute")
def free_endpoint(request: Request):
    return {"message": "Free tier — 10 requests per minute"}

@app.get("/premium")
@limiter.limit("100/minute")
def premium_endpoint(request: Request):
    return {"message": "Premium tier — 100 requests per minute"}

@app.get("/strict")
@limiter.limit("2/minute")
def strict_endpoint(request: Request):
    return {"message": "Strict — only 2 per minute"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8019)

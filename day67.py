# Day 67 — Middleware

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# TOPIC 3: CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# TOPIC 1: Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"Completed: {response.status_code}")
    return response

# TOPIC 2: Request timing middleware
import time

@app.middleware("http")
async def time_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 4)
    response.headers["X-Process-Time"] = str(duration)
    print(f"{request.url.path} took {duration}s")
    return response

@app.get("/hello")
def hello():
    return {"message": "Hello!"}

@app.get("/goodbye")
def goodbye():
    return {"message": "Goodbye!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8025)

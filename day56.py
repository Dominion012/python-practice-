# Day 56 — Background Tasks in FastAPI

from fastapi import FastAPI, BackgroundTasks
import uvicorn
import time

app = FastAPI()

# TOPIC 1: Basic background task
def write_log(message: str):
    time.sleep(2)  # simulates slow work (e.g. saving to a file, sending email)
    with open("log.txt", "a") as f:
        f.write(message + "\n")
    print(f"Logged: {message}")

@app.post("/submit")
def submit(background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, "Someone submitted the form")
    return {"message": "Received! Processing in background."}


# TOPIC 2: Background task with request data
from pydantic import BaseModel

class UserRequest(BaseModel):
    name: str
    email: str

def send_welcome_email(name: str, email: str):
    time.sleep(1)
    print(f"Sending welcome email to {name} at {email}")

@app.post("/register")
def register(request: UserRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(send_welcome_email, request.name, request.email)
    return {"message": f"Welcome {request.name}! Check your email."}


# TOPIC 3: Multiple background tasks
def log_activity(action: str):
    with open("activity.txt", "a") as f:
        f.write(action + "\n")
    print(f"Activity logged: {action}")

def notify_admin(user: str):
    time.sleep(1)
    print(f"Admin notified: new user {user} signed up")

@app.post("/signup")
def signup(request: UserRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(log_activity, f"{request.name} signed up")
    background_tasks.add_task(notify_admin, request.name)
    background_tasks.add_task(send_welcome_email, request.name, request.email)
    return {"message": f"Signed up {request.name}!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)

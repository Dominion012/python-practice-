# JWT Side Topic 2 — JWT in FastAPI (login endpoint)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jwt
import datetime
import uvicorn

app = FastAPI()
SECRET_KEY = "supersecretkey_atleast32charslong!!"

# TOPIC 1: Login endpoint that returns a JWT
class LoginRequest(BaseModel):
    email: str
    password: str

FAKE_USERS = {
    "domi@email.com": "password123",
    "admin@email.com": "admin456"
}

def create_token(email: str):
    payload = {
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

@app.post("/login")
def login(request: LoginRequest):
    stored_password = FAKE_USERS.get(request.email)
    if not stored_password or stored_password != request.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token(request.email)
    return {"access_token": token, "token_type": "bearer"}


# TOPIC 2: Protected endpoint that requires the token
from fastapi import Header, Depends

def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
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
    uvicorn.run(app, host="0.0.0.0", port=8010)

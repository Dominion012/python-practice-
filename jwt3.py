# JWT Side Topic 3 — Token Expiry and Refresh Tokens

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import jwt
import datetime
import uvicorn

app = FastAPI()
SECRET_KEY = "supersecretkey_atleast32charslong!!"

FAKE_USERS = {
    "domi@email.com": "password123"
}

class LoginRequest(BaseModel):
    email: str
    password: str

# TOPIC 1: Two tokens — access (short) and refresh (long)
def create_access_token(email: str):
    payload = {
        "email": email,
        "type": "access",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def create_refresh_token(email: str):
    payload = {
        "email": email,
        "type": "refresh",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

@app.post("/login")
def login(request: LoginRequest):
    stored = FAKE_USERS.get(request.email)
    if not stored or stored != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "access_token": create_access_token(request.email),
        "refresh_token": create_refresh_token(request.email)
    }


# TOPIC 2: Use refresh token to get a new access token
class RefreshRequest(BaseModel):
    refresh_token: str

@app.post("/refresh")
def refresh(request: RefreshRequest):
    try:
        payload = jwt.decode(request.refresh_token, SECRET_KEY, algorithms=["HS256"])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Not a refresh token")
        new_access = create_access_token(payload["email"])
        return {"access_token": new_access}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired, please login again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# TOPIC 3: Protected route using access token
def verify_access_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Not an access token")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Access token expired, use /refresh")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/profile")
def profile(payload: dict = Depends(verify_access_token)):
    return {"email": payload["email"], "message": "Access granted"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)

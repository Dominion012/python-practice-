# JWT Side Topic 4 — OAuth2PasswordBearer (proper JWT in FastAPI)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import jwt
import datetime
import uvicorn

app = FastAPI()
SECRET_KEY = "supersecretkey_atleast32charslong!!"

# TOPIC 1: OAuth2PasswordBearer tells FastAPI/docs where the login endpoint is
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

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


# TOPIC 2: Login using OAuth2PasswordRequestForm (docs sends this automatically)
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    password = FAKE_USERS.get(form_data.username)
    if not password or password != form_data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(form_data.username)
    return {"access_token": token, "token_type": "bearer"}


# TOPIC 3: Protected route — token extracted automatically from header
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/me")
def get_me(payload: dict = Depends(get_current_user)):
    return {"email": payload["email"]}

@app.get("/dashboard")
def dashboard(payload: dict = Depends(get_current_user)):
    return {"message": f"Welcome {payload['email']}!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8014)

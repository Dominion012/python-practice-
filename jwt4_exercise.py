from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import jwt
import datetime
import uvicorn

app = FastAPI()
outh2_scheme = OAuth2PasswordBearer(tokenUrl="/login")
FAKE_USERS = {
    "domi012@email.com" : "password123",
    "flaunty012" : "slyunt012",
    "admin" : "just123"
}
SECRET_KEY = "nnwshnrwjgejje9efj"
def create_token(email:str):
    payload = {
        "email" :email,
        "exp" : datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

@app.post("/login")
def login(form_data:OAuth2PasswordRequestForm=Depends()):
    password = FAKE_USERS.get(form_data.username)
    if not password or password != form_data.password:
        raise HTTPException(status_code=401, detail="Invalid input")
    token = create_token(form_data.username)
    return {"access_token": token, "token_type": "Bearer"}

def get_current_user(token:str=Depends(outh2_scheme)):
    try: 
        payload = jwt.decode(token, SECRET_KEY, algorithms= ["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code= 401, detail= "Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail= "Invalid token")


@app.get("/me")
def get_me(payload:dict=Depends(get_current_user)):
    return {"username" : payload['email']}

@app.get("/admin")
def get_admin(payload:dict=Depends(get_current_user)):
    if payload["email"] == "admin":
        return {"message" : "Welcome admin!"}
    else:
        raise HTTPException(status_code=403, detail= "Acess denied")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8022)

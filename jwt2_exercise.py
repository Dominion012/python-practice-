import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Header, Depends
import jwt
import datetime
FAKE_USERS = {
    "flaunty" : "ratpack012",
    "roaringmrs501" : "blitsrig012"
}
SECRET_KEY = "manijust_love_pythone8976_rararvgs"
app = FastAPI()

class Ent(BaseModel):
    username: str
    password : str 

def create_token(username:str):
    payload = {
        "username" : username,
        "exp" : datetime.datetime.utcnow() + datetime.timedelta(1)
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=["HS256"])

@app.post("/login")
def posty(request:Ent):
    password = FAKE_USERS.get(request.username)
    if not password or password != request.password:
        raise HTTPException(status_code= 401, detail= "Unauthorized user")
    
    token = create_token(request.username)
    return {"Token created" : token}

def verify (authorization:str = Header(None)):
    if not authorization or not authorization.startswith("bearer "):
        raise HTTPException(status_code= 401, detail= "Invalid Token")
    token = authorization.split(" ")[1]
    try: 
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidKeyError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
       
@app.get("/dashboard")
def dashy(payload:dict=Depends(verify)):
    return {"email": payload["username"]}

if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port= 8011)

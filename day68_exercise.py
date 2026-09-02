from fastapi import FastAPI, UploadFile,HTTPException, File, Request
from fastapi.middleware.cors import CORSMiddleware
import time
import os
from fastapi.testclient import TestClient
app = FastAPI()

app.add_middleware (
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

@app.middleware("http")
async def log_req (request: Request, call_next):
    print(f"{request.method} from {request.url.path}")
    response = await call_next(request)
    return response 

@app.middleware("http")
async def tim_req(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round(time.time() - start, 4)
    response.headers["Time_taken"] = str(duration)
    print (f"{request.url.path} time taken{duration}")
    return response 


@app.get("/hello")
def hello():
    return {"message": "hello"}
ALLOWED_FILES = ["image/jpg", "image/png", "image/jpeg"]
MAX_SIZE = 2 * 1024 *1024
UPLOAD_DIR = "profile_uploads"
os.makedirs(UPLOAD_DIR, exist_ok= True)
@app.post("/profile/upload")
async def profile_upload(file:UploadFile=File(...)):
    if file.content_type not in ALLOWED_FILES:
        raise HTTPException (status_code= 400, detail= "File type not allowed")
    
    contents = await file.read()

    if len(contents) > MAX_SIZE:
        raise HTTPException( status_code= 400, detail= " File size exceeds accepted limit")
    
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open (save_path, "wb") as f:
        f.write(contents)

    return {"filename": file.filename, "message": "File uploaded successfully"}

    

@app.get("/profile/files")
def get_uploads():
    return os.listdir("profile_uploads")

client = TestClient(app)

def test_valid_image():
    fake_image = b"fake image bytes"
    response = client.post("/profile/upload", files={"file": ("photo.jpg", fake_image, "image/jpeg")})
    assert response.status_code == 200

def test_invalid_file():
    fake_txt = b"hello"
    response = client.post("/profile/upload", files={"file": ("doc.txt", fake_txt, "text/plain")})
    assert response.status_code == 400

def test_list_files():
    response = client.get("/profile/files")
    assert response.status_code == 200



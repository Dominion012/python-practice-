# Day 68 — File Uploads

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os

app = FastAPI()

# TOPIC 1: Basic file upload
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "size": len(contents),
        "content_type": file.content_type
    }

# TOPIC 2: Validate file type and size
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/gif"]
MAX_SIZE = 1 * 1024 * 1024  # 1MB in bytes

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and GIF images allowed")

    contents = await file.read()

    if len(contents) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 1MB")

    return {"filename": file.filename, "size": len(contents), "content_type": file.content_type}

# TOPIC 3: Save file to disk
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/save")
async def upload_and_save(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only images allowed")

    contents = await file.read()

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    return {"message": f"Saved to {save_path}", "size": len(contents)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8026)

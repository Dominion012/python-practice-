# Day 69 — Deployment

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="Domi's Notes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

notes = {}
counter = 1

class NoteRequest(BaseModel):
    title: str
    content: str

@app.get("/")
def root():
    return {"message": "Notes API is live!"}

@app.post("/notes")
def create_note(request: NoteRequest):
    global counter
    notes[counter] = {"id": counter, "title": request.title, "content": request.content}
    note = notes[counter]
    counter += 1
    return note

@app.get("/notes")
def get_notes():
    return list(notes.values())

@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    if note_id not in notes:
        raise HTTPException(status_code=404, detail="Note not found")
    del notes[note_id]
    return {"message": f"Note {note_id} deleted"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8027))
    uvicorn.run(app, host="0.0.0.0", port=port)

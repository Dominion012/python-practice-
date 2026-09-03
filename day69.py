# Day 69 — Deployment

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, Session
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

engine = create_engine("sqlite:///notes.db")

class Base(DeclarativeBase):
    pass

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(String)

Base.metadata.create_all(engine)

class NoteRequest(BaseModel):
    title: str
    content: str

@app.get("/")
def root():
    return {"message": "Notes API is live!"}

@app.post("/notes")
def create_note(request: NoteRequest):
    with Session(engine) as session:
        note = Note(title=request.title, content=request.content)
        session.add(note)
        session.commit()
        return {"id": note.id, "title": note.title, "content": note.content}

@app.get("/notes")
def get_notes():
    with Session(engine) as session:
        notes = session.query(Note).all()
        return [{"id": n.id, "title": n.title, "content": n.content} for n in notes]

@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    with Session(engine) as session:
        note = session.query(Note).filter(Note.id == note_id).first()
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        session.delete(note)
        session.commit()
        return {"message": f"Note {note_id} deleted"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8027))
    uvicorn.run(app, host="0.0.0.0", port=port)

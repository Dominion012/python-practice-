# SQL Side Topic 5 — FastAPI + SQLAlchemy
from fastapi import FastAPI
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, Session
from pydantic import BaseModel
import uvicorn

engine = create_engine("sqlite:///api.sqlite")

class Base(DeclarativeBase):
    pass

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(String)

Base.metadata.create_all(engine)

app = FastAPI()

class NoteRequest(BaseModel):
    title: str
    content: str

@app.post("/notes")
def create_note(request: NoteRequest):
    with Session(engine) as session:
        note = Note(title=request.title, content=request.content)
        session.add(note)
        session.commit()
        session.refresh(note)
        return {"id": note.id, "title": note.title, "content": note.content}

@app.get("/notes")
def get_notes():
    with Session(engine) as session:
        notes = session.query(Note).all()
        return [{"id": n.id, "title": n.title, "content": n.content} for n in notes]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)

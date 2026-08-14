from fastapi import FastAPI
from sqlalchemy import create_engine, Column, INTEGER, String,ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, relationship
from pydantic import BaseModel
import uvicorn

app = FastAPI()
engine = create_engine("sqlite:///api.sqlite")

class Base(DeclarativeBase):
    pass

class Author(Base):
    __tablename__ = "Author"
    id = Column(INTEGER, primary_key= True)
    name = Column(String)
    country = Column(String)
    books = relationship("Books", back_populates= "author")

class EntryPost(BaseModel):
    name:str
    title: str
    country: str
    genre: str

class UpdateFields(BaseModel):
    book_id:int
    title:str
    genre: str

class Books(Base):
    __tablename__ = "Book"
    id = Column(INTEGER, primary_key=True)
    title = Column(String)
    genre = Column(String)
    author_id = Column(INTEGER, ForeignKey("Author.id"))
    author = relationship("Author", back_populates="books")

Base.metadata.create_all(engine)

@app.post("/create")
def create(request:EntryPost):
    with Session(engine) as session:
        exist = session.query(Author).filter(Author.name ==request.name).first()
        if exist:
           book_new = Books(title = request.title, genre = request.genre, author = exist)
           session.add(book_new)
           session.commit()
           session.refresh(book_new)
           return {"Author name" : exist.name, "book": book_new.title}
        user = Author(name = request.name, country = request.country)
        book1 = Books(title = request.title, genre = request.genre ,author = user )
        session.add_all([user, book1])
        session.commit()
        session.refresh(user)
        session.refresh(book1)
        return {"Author name" : user.name, "book": book1.title}

@app.put("/update")
def update(request:UpdateFields):
    with Session(engine) as session:
        valid = session.query(Books).filter(Books.id == request.book_id).first()
        if not valid:
            return {"message" : "no matching entries"}
        valid.title = request.title
        valid.genre = request.genre
        session.commit()
        return {"id": valid.id, "title": valid.title, "genre": valid.genre}
        
@app.get("/all")
def get_all():
    with Session(engine) as session:
        books = session.query(Books).all()
        
        return [{"Title":b.title, "Genre":b.genre, "Author": b.author.name} for b in books]

@app.delete("/delete/{book_id}")
def delete(book_id:int):
    with Session(engine) as session:
        valid = session.query(Books).filter(Books.id == book_id).first()
        if not valid:
            return {"message" : "Book not found"}
        
        book_title = valid.title 
        session.delete(valid)
        session.commit()
        return{"message": f"Book {book_title} deleted"}
    
@app.get("/genre_filter")
def g_filter(type:str):
    with Session(engine) as session:
        valid = session.query(Books).filter(Books.genre.like(f"%{type}%")).order_by(Books.title)
        return [{"id" : v.id, "title": v.title, "genre": v.genre, "author": v.author.name} for v in valid ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
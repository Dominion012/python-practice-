# SQL Side Topic 2 — CRUD
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, Session

engine = create_engine("sqlite:///mydb.sqlite")

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)

Base.metadata.create_all(engine)

# CREATE
with Session(engine) as session:
    user = User(name="Domi", email="domi@example.com")
    session.add(user)
    session.commit()
    print("Created:", user.name)

# READ
with Session(engine) as session:
    users = session.query(User).all()
    for u in users:
        print("Read:", u.id, u.name, u.email)

# UPDATE
with Session(engine) as session:
    user = session.query(User).filter(User.name == "Domi").first()
    user.email = "updated@example.com"
    session.commit()
    print("Updated email:", user.email)

# DELETE
with Session(engine) as session:
    user = session.query(User).filter(User.name == "Domi").first()
    session.delete(user)
    session.commit()
    print("Deleted:", user.name)

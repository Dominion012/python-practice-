# SQL Side Topic 4 — Queries: filter, order, limit
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, relationship

engine = create_engine("sqlite:///mydb.sqlite")

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "User"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)

# Filter — find specific records
with Session(engine) as session:
    domi = session.query(User).filter(User.name == "Domi").first()
    print("Filter:", domi.name if domi else "Not found")

# All matching — filter without .first()
with Session(engine) as session:
    results = session.query(User).filter(User.email.like("%gmail%")).all()
    print("All gmail users:", [u.name for u in results])

# Order by
with Session(engine) as session:
    users = session.query(User).order_by(User.name).all()
    print("Ordered:", [u.name for u in users])

# Limit
with Session(engine) as session:
    users = session.query(User).limit(1).all()
    print("First only:", [u.name for u in users])

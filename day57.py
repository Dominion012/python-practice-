# Day 57 — PostgreSQL with SQLAlchemy

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, Session

# TOPIC 1: Connect to PostgreSQL instead of SQLite
# SQLite:    "sqlite:///mydb.sqlite"
# PostgreSQL: "postgresql://user:password@host/dbname"
import os
username = os.environ.get("USER", "postgres")
engine = create_engine(f"postgresql://{username}@localhost/mydb")

class Base(DeclarativeBase):
    pass

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Integer)

Base.metadata.create_all(engine)
print("Table created in PostgreSQL")


# TOPIC 2: CRUD — same as SQLite, nothing changes
with Session(engine) as session:
    existing = session.query(Product).filter(Product.name == "Laptop").first()
    if not existing:
        product = Product(name="Laptop", price=1200)
        session.add(product)
        session.commit()
        print("Product added")
    else:
        print("Product already exists")


# TOPIC 3: Query
with Session(engine) as session:
    products = session.query(Product).all()
    for p in products:
        print(f"{p.id} - {p.name}: ${p.price}")

# Day 58 — Alembic Database Migrations
#
# Alembic workflow:
# 1. Change your model
# 2. python3 -m alembic revision --autogenerate -m "description"
# 3. python3 -m alembic upgrade head
#
# Useful commands:
# python3 -m alembic history        — see all migrations
# python3 -m alembic current        — see current version
# python3 -m alembic downgrade -1   — undo last migration

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import DeclarativeBase, Session
import os

username = os.environ.get("USER", "postgres")
engine = create_engine(f"postgresql://{username}@localhost/mydb")

class Base(DeclarativeBase):
    pass

# After running alembic, you can safely add new columns to existing models.
# Add 'category' below, then run: python3 -m alembic revision --autogenerate -m "add category to products"
# Then: python3 -m alembic upgrade head
class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Integer)
    category = Column(String)  # new column — added without losing existing data

Base.metadata.create_all(engine)

with Session(engine) as session:
    products = session.query(Product).all()
    for p in products:
        print(f"{p.name} - ${p.price} - {p.category}")

# SQL Side Topic 3 — Relationships and Foreign Keys
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, relationship

engine = create_engine("sqlite:///mydb.sqlite")

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", back_populates="posts")

Base.metadata.create_all(engine)

with Session(engine) as session:
    user = User(name="Domi")
    post1 = Post(title="My first post", author=user)
    post2 = Post(title="My second post", author=user)
    session.add_all([user, post1, post2])
    session.commit()

with Session(engine) as session:
    user = session.query(User).filter(User.name == "Domi").first()
    print(f"User: {user.name}")
    for post in user.posts:
        print(f"  Post: {post.title}")

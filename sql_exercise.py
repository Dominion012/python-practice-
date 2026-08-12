from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session,relationship

engine = create_engine("sqlite:///mydb.sqlite")

class base(DeclarativeBase):
    pass

class user(base):
    __tablename__ = "User"
    id = Column(Integer, primary_key=True )
    name = Column(String)
    email = Column(String)
    articles = relationship("articles", back_populates="author")

class articles(base):
    __tablename__ = "Articles"
    title = Column(String)
    id = Column(Integer, primary_key=True)
    content = Column(String)
    author_id = Column(Integer, ForeignKey("User.id"))
    author = relationship("user", back_populates="articles")

base.metadata.create_all(engine)

with Session(engine) as session:
    entry = user(name = "Domi", email = "koladedomin012@gmail.com")
    entry2 = user(name = "Alaba", email = "faje012@gmail.com")
    article1  = articles(title = " My first exercise", content = "still deciding", author = entry)
    article2 = articles(title = "My second post", content = "how to learn python", author = entry)
    article3 = articles(title = "Faje's little story", content = "How to trek from ijeda to iloko", author = entry2)
    session.add_all([article1,article2,article3])
    session.commit()

# with Session(engine) as session:
#     users = session.query(user).all()
#     for u in users:
#         for p in u.articles:
#             print(f"Author : {u.name}\nArticle: {p.content}")

with Session(engine) as session:
    users = session.query(articles).filter(articles.title == " My first exercise").first()
    users.title = "Changed"
    session.commit()

with Session(engine) as session:
    users = session.query(articles).filter(articles.title == "My second post").first()
    session.delete(users)
    session.commit()

with Session(engine) as session:
    users = session.query(user).all()
    for u in users:
        for p in u.articles:
            print(f"Author : {u.name}\nTitle: {p.title}\nArticle: {p.content}")

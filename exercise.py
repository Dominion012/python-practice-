import json 
import os 

class Book :
    def __init__(self, title, author , pages , is_read):
        self.title = title 
        self.author = author 
        self.pages = pages 
        self.is_read = is_read

    def describe(self):
        print(f" The title is {self.title} by {self.author} and it has {self.pages} pages ")

    def mark_as_read(self):
        self.is_read = True

      


class Ebook (Book):
    def __init__(self, title, author, pages, file_size_mb, is_read):
        super().__init__(title, author, pages, is_read)
        self.file_size_mb = file_size_mb
    
    def describe(self):
        print(f" The title is {self.title} by {self.author} it has {self.pages} pages and a file size of {self.file_size_mb} ")


class library :
    def __init__(self):
        self.books = []
        self.load()

    def add_book(self, title, author, pages, is_read): 
        if pages < 1 :
            raise ValueError ("Pages cannot be less than one")
        
        add = Book(title, author, pages, is_read)
        self.books.append(add)
        self.save()
    
    def load(self):
       if not os.path.exists("books.json"):
        return
       with open("books.json", "r") as file:
        content = json.load(file)
        for item in content:
            book = Book(item["title"], item["author"], item["pages"], item["is_read"])
            self.books.append(book)

    
    def save(self):
        data = []
        for books in self.books:
            data.append({"title":books.title, "author": books.author, "pages": books.pages, "is_read":books.is_read})
        with open ("books.json","w") as file :
            json.dump(data, file, indent= 3)

    def show_unread(self):
        for books in self.books :
            if books.is_read ==  False:
                books.describe()
            else:
                return 

    def show_all(self):
        if self.books == []:
            return
        for books in self.books:
            books.describe()


my_library = library()
my_library.add_book("The Alchemist", "Paulo Coelho", 208, False)
my_library.add_book("Atomic Habits", "James Clear", 320, True)
my_library.add_book("Python Crash Course", "Eric Matthes", 544, False)

print("--- All Books ---")
my_library.show_all()

print("--- Unread Books ---")
my_library.show_unread()

try:
    my_library.add_book("Bad Book", "Nobody", 0, False)
except ValueError as e:
    print(f"Error: {e}")


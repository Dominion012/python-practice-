# Day 5 - Object Oriented Programming (OOP)
# Classes, Objects, __init__, Instance Variables, Inheritance, Encapsulation


class Car:
    def __init__ (self, make,model ,year):
        self.make = make 
        self.model = model
        self.year = year
    def describe (self):
        print(f"This is a {self.year} {self.make} {self.model}")

    def age (self):
        return 2026 - self.year
    
    def is_vintage(self):
        return 2026 - self.year > 25
            
        


car1 = Car("Ford","Taurus", 2026)
car2 = Car("Mercedez","c-class", 1999)
car3 = Car("Mazda","cx90", 2019)
#print(car1.make, car1.model, car1.year)
#print(car2.make, car2.model,car2.year)
#print(car3.make,car3.model, car3.year)

#car1.describe()
#car2.describe()
#car3.describe()
#print(car2.is_vintage())
#print(car1.is_vintage())
#print(car3.is_vintage())

# Inheritance 

class Truck (Car):
    def __init__ (self , make , model , year , payload_tons):
        super().__init__(make , model , year)
        self.payload_tons = payload_tons

    def haul_info (self):
        print(f'This truck can carry {self.payload_tons} tons')

my_truck = Truck("Dodge", "Ram", 2024, 5 )
#my_truck.describe()
#my_truck.haul_info()

   
class Bankaccount:
    def __init__ (self, owner, balance ):
        self.owner = owner 
        self._balance = balance 
    
    def get_balance (self):
        return self._balance 

    def deposit (self , amount):
        self._balance += amount
    
    def withdraw (self, amount):
        if self._balance >= amount:
            self._balance -= amount
            
        
        else:
            raise ValueError ("Insufficient funds")
        
        
acc = Bankaccount("Domi", 950)
acc.deposit(200)
acc.withdraw(1600)
print(acc.get_balance())

        
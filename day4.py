# Catching value errors 
#try:
  #number = int("hello")
  #print(number)
#except ValueError:
  #print("This is not a valid number.")

# Using try with else and finally (multiple error types handled)
#try:
    #num = int(input("Enter a number: "))
    #div = 100 / num 
   
#except ValueError:
   #print("This is not a valid number.")
#except ZeroDivisionError:
   #print("Can't divide by zero")
#else:
   #print (f'100 divided by your number is : {div}')
#finally:
   #print("Done.")


def get_age(age):
   age = int(age)
   if age < 0:
      raise ValueError ("Age cannot be negative")
   return age 
try:
    print(get_age(25))
except ValueError as e:
   print(f"Error:{e}")

try:
   print(get_age("-5"))
except ValueError as e:
   print(f"Error: {e}")
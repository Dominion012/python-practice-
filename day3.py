import os
import json
#if os.path.exists("diary.txt"):
    #print("Found it!")
#else:
    #print("Not found")

#path = os.path.join("python-practice", "dairy.txt")
#print(path)
#print(__file__)
#print(os.path.dirname(__file__))
# Writing files 
#with open ("diary.txt", "w" ) as file:
    #file.write("My name is Dominion\n")
    #file.write("I am learning python on day 3")

#Appending to files
#with open ("diary.txt", "a") as file :
    #file.write("This line was added later.\n")
    #file.write("And this is the third added line.\n")

# Reading files
#with open("diary.txt", "r") as file :
    #content = file.read()
    #print(content)

#writing JSON

#contacts = [{"name": "Dominion", "phone":"8623475066" },
            #{"name": "Claude", "phone": "8654009867"}]
#with open("contacts.json", "w") as file :
    #json.dump(contacts,file)
#with open ("contacts.json", "r") as file:
   # loaded = json.load(file)
   # print(loaded)

#for contact in contacts:
  #  if contact["name"] == "Claude":
    #    contact["phone"] = "000000000"

#with open("contacts.json", "w") as file:
   # json.dump(contacts,file)

#for contact in contacts:
   # print(contact)

try:
    with open("fake.json", "r") as file:
        loaded = json.load(file)
except FileNotFoundError:
    print('File does not exist')
except json.JSONDecodeError:
    print("File exixsts but the JSON is broken")
   

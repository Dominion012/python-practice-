import json
import os 
File = "contacts.json"

def load_contacts():
    if not os.path.exists(File):
        return []
    
    with open (File, "r") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return []

def save_contacts(contacts):
    with open (File, "w") as file:
        json.dump(contacts, file)

def add_contacts(name, phone):
    contacts = load_contacts()
    if name == ""or phone == "":
        raise ValueError("Name or phone field empty.")
    else:
     contacts.append({"name": name, "phone":phone})
     save_contacts(contacts)
     print(f"{name} added.")

def view_contacts():
    contacts = load_contacts()
    if not contacts:
        print("No contacts yet")
    for contact in contacts:
        print(f"{contact['name']} - {contact['phone']}")

def update_contact(name, new_phone):
    contacts = load_contacts()
    found = False
    for contact in contacts:
        if contact['name'] == name:
            contact['phone'] = new_phone
            found = True
            break
    if not found:
        raise ValueError("Contact not found.")
    save_contacts(contacts)

def delete_contact(name):
    contacts = load_contacts()
    found = False 
    for contact in contacts:
        if contact["name"] == name:
            contacts.remove(contact)
            found = True 
            break
    if not found:
        raise ValueError ("Contact not found.")
    save_contacts(contacts)
    print(f"{name} deleted")
 


#add_contacts("Dominion", "8635674056")
#add_contacts("Claude", "8078675043")
#try:
  #add_contacts("sekemi", "89099868")
#except ValueError as e:
    #print(f"Error: {e}")

#try :
    #update_contact("sekemi", "111111111")
#except ValueError as e:
   # print(f"Error {e}")


try :
    delete_contact("sekem")
except ValueError as e:
    print(f"Error {e}")

try :
    delete_contact('sekemi')
except ValueError as e:
    print(f"Error {e}")


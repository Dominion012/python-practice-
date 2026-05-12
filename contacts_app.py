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
    for contact in contacts:
      if contact['name'] == name:
        contact['phone'] = new_phone
    save_contacts(contacts)

def delete_contact(name):
    contacts = load_contacts()
    for contact in contacts:
        if contact["name"] == name:
            contacts.remove(contact)
            break
    save_contacts(contacts)
    print(f"{name} deleted.")


#add_contacts("Dominion", "8635674056")
#add_contacts("Claude", "8078675043")

update_contact("Dominion", "11111111")
delete_contact("Claude")
view_contacts()

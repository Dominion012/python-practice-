import requests

BASE_URL = 'http://127.0.0.1:5001/predict'

# Passenger 1: young, first-class, female, high fare — likely survivor
passenger_1 = {'Pclass': 1, 'Sex': 0, 'Age': 24, 'SibSp': 0, 'Parch': 0, 'Fare': 80}

# Passenger 2: older, third-class, male, low fare — historically much lower survival rate
passenger_2 = {'Pclass': 3, 'Sex': 1, 'Age': 40, 'SibSp': 0, 'Parch': 0, 'Fare': 7}

for i, passenger in enumerate([passenger_1, passenger_2], start=1):
    response = requests.post(BASE_URL, json=passenger)
    print(f"Passenger {i}: {response.json()}")

# Bad request: missing the Age field
response = requests.post(BASE_URL, json={
    'Pclass': 1, 'Sex': 0, 'SibSp': 0, 'Parch': 0, 'Fare': 80
})
print(f"Missing field: {response.status_code} {response.json()}")

# Bad request: non-numeric value
response = requests.post(BASE_URL, json={
    'Pclass': 1, 'Sex': 0, 'Age': 'twenty-two', 'SibSp': 0, 'Parch': 0, 'Fare': 80
})
print(f"Non-numeric field: {response.status_code} {response.json()}")

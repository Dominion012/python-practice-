# JWT Side Topic 1 — What is JWT and how it works
import jwt
import datetime

SECRET_KEY = "mysecretkey"

# TOPIC 1: Create a token
def create_token(user_id: int, email: str):
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

token = create_token(1, "domi@email.com")
print("Token:", token)


# TOPIC 2: Decode a token
def decode_token(token: str):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return payload

data = decode_token(token)
print("Decoded:", data)


# TOPIC 3: What happens with a wrong key
try:
    fake = jwt.decode(token, "wrongkey", algorithms=["HS256"])
except jwt.InvalidTokenError as e:
    print("Invalid token:", e)


# TOPIC 4: What happens when token is expired
expired_payload = {
    "user_id": 2,
    "email": "test@email.com",
    "exp": datetime.datetime.utcnow() - datetime.timedelta(hours=1)  # already expired
}
expired_token = jwt.encode(expired_payload, SECRET_KEY, algorithm="HS256")

try:
    jwt.decode(expired_token, SECRET_KEY, algorithms=["HS256"])
except jwt.ExpiredSignatureError as e:
    print("Expired token:", e)

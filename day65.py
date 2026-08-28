# Day 65 — WebSockets

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import jwt
import uvicorn

SECRET_KEY = "supersecretkey_atleast32charslong!!"

app = FastAPI()

# TOPIC 1: Basic WebSocket connection
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        await websocket.send_text(f"Echo: {message}")

# TOPIC 2: Broadcasting to multiple clients
connected_clients = []

@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            for client in connected_clients:
                await client.send_text(f"Broadcast: {message}")
    except:
        connected_clients.remove(websocket)

# TOPIC 3: WebSocket + JWT auth
@app.websocket("/secure")
async def secure_chat(websocket: WebSocket, token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload["email"]
    except jwt.InvalidTokenError:
        await websocket.close(code=1008)  # 1008 = policy violation
        return

    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            await websocket.send_text(f"{email}: {message}")
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8024)

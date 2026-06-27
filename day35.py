# Day 35 — Streamlit Frontend for the RAG Chatbot
# Week 7 Day 5

import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:5002"

st.title("Acme Corp Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.sidebar.header("Knowledge Base")
for doc in requests.get(f"{BACKEND_URL}/documents").json()["documents"]:
    st.sidebar.write(f"- {doc}")

st.sidebar.header("Upload a Document")
uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type="txt")
if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post(f"{BACKEND_URL}/upload", files=files)
    st.sidebar.success(response.json()["message"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Ask a question...")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    response = requests.post(f"{BACKEND_URL}/ask", json={
        "question": user_question,
        "session_id": st.session_state.session_id
    })
    data = response.json()
    st.session_state.session_id = data["session_id"]
    answer = data["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

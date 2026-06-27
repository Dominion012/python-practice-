import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from day33 import RAGChatbot, get_or_build_store, chunk_by_paragraph, save_store, model, client

app = Flask(__name__)
CORS(app)   # permissive for dev; lock to the real frontend's domain in production

doc_store = get_or_build_store(model)
sessions = {}   # session_id -> RAGChatbot, keeps concurrent users' conversations isolated

def get_chatbot(session_id):
    if session_id not in sessions:
        sessions[session_id] = RAGChatbot(doc_store, client)
    return sessions[session_id]


@app.route("/")
def home():
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    session_id = data.get("session_id") or str(uuid.uuid4())
    chatbot = get_chatbot(session_id)
    answer = chatbot.ask(data["question"])
    return jsonify({"answer": answer, "session_id": session_id})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".txt"):
        return jsonify({"error": "Only .txt files supported"}), 400

    filepath = os.path.join("docs", file.filename)
    file.save(filepath)

    with open(filepath) as f:
        content = f.read()
    chunks = chunk_by_paragraph(content)
    doc_store.add(chunks, source=file.filename)
    save_store(doc_store)   # so the new doc survives a restart too

    return jsonify({"message": f"Uploaded and indexed {file.filename}", "chunks_added": len(chunks)}), 200


@app.route("/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": sorted(set(doc_store.sources)), "total_chunks": len(doc_store.chunks)})


if __name__ == "__main__":
    app.run(debug=True, port=5002)

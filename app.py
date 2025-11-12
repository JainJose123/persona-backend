import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY", "")
MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")


memory = {
    "xp": 500,
    "last_chat": None,
    "last_tasks": None,
    "history": {
        "chats": [],   # list of {user, ai, ts}
        "tasks": [],   # list of {goal, items (string), ts}
        "emails": []   # list of {thread, draft, ts}
    }
}

def _headers(title="Persona Assistant"):
    return {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "HTTP-Referer": "http://localhost:5001",
        "X-Title": title,
        "Content-Type": "application/json",
    }


def call_openrouter(messages, model=MODEL, max_tokens=120, temperature=0.7, title="Persona"):
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=_headers(title),
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        data = r.json()
        return data
    except Exception as e:
        return {"error": str(e)}


@app.route("/api/ask", methods=["POST"])
def ask():
    """Concise chat; store user + AI reply in history."""
    data = request.get_json(force=True)
    q = data.get("message", "").strip()
    if not q:
        return jsonify({"error": "Empty message"}), 400

    memory["last_chat"] = q

    system = (
        "You are Persona, a concise, friendly assistant. "
        "Reply in under 3 lines, with clarity and encouragement."
    )

    models = [
        "meta-llama/llama-3.1-8b-instruct",   # primary
        "mistralai/mistral-7b-instruct-v0.3", # fallback 1
        "google/gemma-2-9b-it"                # fallback 2
    ]

    for m in models:
        resp = call_openrouter(
            messages=[{"role":"system","content":system},{"role":"user","content":q}],
            model=m, max_tokens=80, temperature=0.7, title="Persona Chat"
        )
        if "choices" in resp:
            ai = resp["choices"][0]["message"]["content"]
            # store history
            memory["history"]["chats"].append({
                "user": q, "ai": ai, "ts": __import__("time").time()
            })
            return jsonify({"reply": ai, "model": m})

    return jsonify({"error":"All models failed"}), 500


@app.route("/api/tasks", methods=["POST"])
def create_tasks():
    """Generate 3 short actionable bullets; store in history."""
    data = request.get_json(force=True)
    goal = data.get("goal", "Plan my day effectively").strip()

    system = (
        "You are Persona, an intelligent planner. "
        "Output exactly 3 short bullet points, crisp and doable."
    )
    resp = call_openrouter(
        messages=[{"role":"system","content":system},{"role":"user","content":goal}],
        model=MODEL, max_tokens=90, temperature=0.7, title="Persona Task Generator"
    )
    if "choices" in resp:
        items = resp["choices"][0]["message"]["content"]
        memory["last_tasks"] = items
        memory["history"]["tasks"].append({
            "goal": goal, "items": items, "ts": __import__("time").time()
        })
        return jsonify({"tasks": items})
    return jsonify({"error": resp}), 400


@app.route("/api/draft-email", methods=["POST"])
def draft_email():
    """Generate a short professional email reply; store in history."""
    data = request.get_json(force=True)
    thread = data.get("thread", "").strip()
    if not thread:
        return jsonify({"error": "Empty thread"}), 400

    memory["last_chat"] = thread

    system = (
        "You are Persona, a professional email assistant. "
        "Write a concise reply under 100 words, with greeting and closing."
    )
    resp = call_openrouter(
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":f"Draft a reply for this thread:\n\n{thread}"}
        ],
        model=MODEL, max_tokens=130, temperature=0.6, title="Persona Email Writer"
    )
    if "choices" in resp:
        draft = resp["choices"][0]["message"]["content"]
        memory["history"]["emails"].append({
            "thread": thread, "draft": draft, "ts": __import__("time").time()
        })
        return jsonify({"draft": draft})
    return jsonify({"error": resp}), 400


@app.route("/api/xp", methods=["POST"])
def xp_update():
    """Add XP based on action type and return total."""
    data = request.get_json(force=True)
    action = (data.get("action") or "chat").lower()
    gain = {"chat":10, "task":20, "email":15}.get(action, 5)
    memory["xp"] += gain
    return jsonify({"action": action, "xp_gained": gain, "total_xp": memory["xp"]})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "model": MODEL,
        "xp": memory["xp"],
        "last_chat": memory["last_chat"],
        "last_tasks": memory["last_tasks"]
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return limited recent history for sidebar."""
    # Return latest 20 of each to keep payload light
    h = memory["history"]
    return jsonify({
        "chats": h["chats"][-20:],
        "tasks": h["tasks"][-20:],
        "emails": h["emails"][-20:]
    })

@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """Clear all history (useful for demo reset)."""
    memory["history"] = {"chats": [], "tasks": [], "emails": []}
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("🚀 Persona Backend running → http://127.0.0.1:5001")
    if not OPENROUTER_KEY:
        print("⚠️  OPENROUTER_KEY is not set. Set it and restart for LLM calls.")
    app.run(host="0.0.0.0", port=5001, debug=True)

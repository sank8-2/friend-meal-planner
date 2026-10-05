#!/usr/bin/env python3
"""Friend Meal Planner — local-first weekly meal planning powered by open-weight Gemma.

Zero third-party dependencies (stdlib only). Serves the web UI and proxies
chat requests to a local Ollama instance running an open-weight Gemma model.
No data ever leaves the machine: no accounts, no API keys, no cloud.

Env:
  PORT              HTTP port to serve on (default 8000)
  OLLAMA_HOST       Ollama base URL (default http://localhost:11434)
  OLLAMA_MODEL      Model name (default gemma3:1b)
  OLLAMA_NUM_GPU    GPU layers for inference (default 0 = CPU, most portable)
"""
import json
import os
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_NUM_GPU = int(os.environ.get("OLLAMA_NUM_GPU", "0"))

SYSTEM_PROMPT = (
    "You are a practical weekly meal planner for Indian home cooking. "
    "Always respect the person's allergies, dislikes, diet and weekly budget strictly — "
    "never suggest anything containing an allergen. Keep recipes simple (30 min or less), "
    "use ingredients common in Bengaluru markets, and give costs in Indian Rupees (₹). "
    "Reply in Markdown with exactly these sections: "
    "## 7-Day Plan (each day: Breakfast, Lunch, Dinner as bullet points), "
    "## Grocery List (grouped: Vegetables, Dairy & Protein, Pantry), "
    "## Budget Check (estimated total vs the given budget, one line per day is too much — one total plus 2 saving tips)."
)

# Deterministic safety net: small models sometimes slip an allergen past the
# prompt, so every plan is scanned line-by-line before it reaches the user.
ALLERGEN_EXPANSIONS = {
    "lactose": ["milk", "curd", "yogurt", "yoghurt", "dahi", "paneer", "ghee",
                "butter", "cheese", "cream", "whey", "khoa", "chaas", "lassi"],
    "dairy": ["milk", "curd", "yogurt", "yoghurt", "dahi", "paneer", "ghee",
              "butter", "cheese", "cream", "whey"],
    "peanut": ["peanut", "groundnut", "moong phali"],
    "gluten": ["wheat", "atta", "maida", "bread", "pasta", "noodles", "suji", "rava"],
    "egg": ["egg", "omelette", "omelet"],
    "soy": ["soy", "tofu"],
}


def allergy_check(plan: str, allergies_raw: str):
    """Return list of (line, matched_keyword) for lines that may contain an allergen."""
    keywords = set()
    for token in allergies_raw.lower().replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        keywords.add(token)
        keywords.update(ALLERGEN_EXPANSIONS.get(token, []))
    if not keywords:
        return []
    flagged = []
    for line in plan.splitlines():
        low = line.lower()
        hits = sorted({k for k in keywords if k and k in low})
        if hits and line.strip().startswith(("*", "-", "#")):
            flagged.append((line.strip()[:160], hits))
    return flagged


def build_prompt(profile: dict) -> str:
    name = profile.get("name", "my friend").strip() or "my friend"
    diet = profile.get("diet", "Vegetarian")
    allergies = profile.get("allergies", "").strip() or "none"
    dislikes = profile.get("dislikes", "").strip() or "none"
    budget = profile.get("budget", "").strip() or "2000"
    days = profile.get("days", "7")
    goal = profile.get("goal", "").strip() or "eat healthy on a budget"
    return (
        f"Create a {days}-day meal plan for {name}.\n"
        f"Diet: {diet}\n"
        f"Allergies (MUST avoid completely): {allergies}\n"
        f"Dislikes: {dislikes}\n"
        f"Weekly grocery budget: ₹{budget}\n"
        f"Goal: {goal}\n"
        f"Keep the whole reply under 550 words."
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "FriendMealPlanner/1.0"

    def log_message(self, *args):  # quieter logs
        pass

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes())
        elif self.path == "/api/health":
            try:
                with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
                    tags = json.loads(r.read().decode())
                models = [m.get("name", "") for m in tags.get("models", [])]
                ok = any(OLLAMA_MODEL in m for m in models)
                self._send(200, json.dumps(
                    {"ok": ok, "model": OLLAMA_MODEL, "available": models}
                ).encode(), "application/json")
            except Exception as e:  # noqa: BLE001
                self._send(200, json.dumps({"ok": False, "error": str(e)}).encode(),
                           "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/plan":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            profile = json.loads(self.rfile.read(length).decode() or "{}")
        except Exception:  # noqa: BLE001
            self._send(400, b"invalid JSON", "text/plain")
            return

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "stream": True,
            "options": {"num_predict": 1100, "temperature": 0.7,
                        "num_gpu": OLLAMA_NUM_GPU},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(profile)},
            ],
        }).encode()

        try:
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/chat", data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            upstream = urllib.request.urlopen(req, timeout=600)
        except Exception as e:  # noqa: BLE001
            self._send(502, f"ollama unreachable: {e}".encode(), "text/plain")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        full = []
        try:
            for line in upstream:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode())
                except Exception:  # noqa: BLE001
                    continue
                text = (chunk.get("message") or {}).get("content", "")
                if text:
                    full.append(text)
                    self.wfile.write(text.encode())
                    self.wfile.flush()
                if chunk.get("done"):
                    break
        finally:
            upstream.close()

        # Deterministic allergy guardrail, appended after the stream.
        flagged = allergy_check("".join(full),
                                profile.get("allergies", "") if isinstance(profile, dict) else "")
        if isinstance(profile, dict) and profile.get("allergies", "").strip():
            if flagged:
                report = "\n\n## ⚠️ Allergy Check — please review these lines\n"
                for line, hits in flagged:
                    report += f"- May contain **{', '.join(hits)}**: {line}\n"
                report += "_The small local model can slip — swap these dishes before cooking._\n"
            else:
                report = ("\n\n## ⚠️ Allergy Check\n"
                          "_No allergen keywords detected in this plan. Still give it a quick read._\n")
            try:
                self.wfile.write(report.encode())
                self.wfile.flush()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    print(f"Serving Friend Meal Planner on http://localhost:{PORT}")
    print(f"Ollama: {OLLAMA_HOST} model={OLLAMA_MODEL} num_gpu={OLLAMA_NUM_GPU}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

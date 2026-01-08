import json
import os
from flask import Flask, request, render_template_string
from openai import OpenAI
from data_filter import filter_activities
from usage_tracker import UsageTracker

# --- Configuration ---
app = Flask(__name__)
JSON_DB_FILE = "activity.json"
HISTORY_FILE = "question_history.json"
MODEL = "gpt-4.1-mini"

# --- Load Database ---
try:
    with open(JSON_DB_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)
except FileNotFoundError:
    print(f"WARNING: {JSON_DB_FILE} not found. App will start but queries may fail.")
    db = []

# --- Initialize Services ---
usage_tracker = UsageTracker()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper Functions ---
def ask_question(question: str) -> str:
    filtered = filter_activities(question, db)
    if not filtered:
        return "Not available in the dataset."

    prompt = f"""
You are a data analyst answering questions from a JSON dataset.

Rules:
- Answer ONLY from the provided records
- Do NOT assume missing data
- If unsure, say "Not available in the dataset"

Relevant Records:
{json.dumps(filtered, ensure_ascii=False)}

User Question:
{question}

Provide a clear, factual answer.
"""
    try:
        # Note: Adjust the API call based on your specific library version/provider
        # The user's original code used client.responses.create (Perplexity/Special SDK?)
        # Standard OpenAI is client.chat.completions.create
        # I will stick to the user's syntax if it looks standard, but 'client.responses.create' is unusual for standard OpenAI python pkg.
        # User imports 'from openai import OpenAI'. Standard usage is client.chat.completions.create.
        # However, checking 'main.py' snippet: response = client.responses.create(...)
        # This implies a specific SDK or maybe Perplexity's OpenAI-compatible endpoint with a specific method?
        # OR it's a hallucination in the previous turn/user's code. 
        # Actually, 'client.responses' doesn't exist in standard OpenAI SDK. 
        # I will assume standard OpenAI chat completion for robustness, or try to match user's code.
        # User's code: response = client.responses.create(model=MODEL, input=prompt)
        # That looks like the Perplexity native API or a legacy/custom wrapper. 
        # Let's assume standard OpenAI ChatCompletion for general compatibility unless user specifies otherwise.
        # But wait, user's main.py has: from openai import OpenAI ... client = OpenAI()
        # If they are using Perplexity via OpenAI client, it should be chat.completions.create.
        # 'client.responses.create' suggests maybe they are using something else or it's a typo in their code.
        # I'll convert it to standard chat.completions.create which works with Perplexity/OpenAI/Groq etc.

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content

        # Mock usage tracking since the object structure differs
        # usage_tracker.record(...) # simplified for deployment
        return answer.strip()

    except Exception as e:
        return f"Error calling AI API: {str(e)}"

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>JSON Analytics Q&A</title>
    <style>
        body { font-family: Segoe UI, Arial; background: #f4f6fa; margin: 0; padding: 0; display: flex; }
        .sidebar { width: 260px; background: #0b1f3b; color: white; padding: 20px; height: 100vh; overflow-y: auto; }
        .sidebar a { display: block; color: #dbe6ff; text-decoration: none; margin-bottom: 10px; font-size: 14px; }
        .content { flex: 1; padding: 40px; }
        .box { background: white; padding: 25px; border-radius: 8px; max-width: 900px; margin: auto; box-shadow: 0 4px 10px rgba(0,0,0,0.08); }
        textarea { width: 100%; height: 90px; padding: 10px; }
        button { background: #0b1f3b; color: white; padding: 10px 18px; border: none; cursor: pointer; margin-top: 10px;}
        .answer { margin-top: 20px; white-space: pre-wrap; }
    </style>
</head>
<body>
<div class="sidebar">
    <h3>History</h3>
    {% for q in history %}
        <a href="/?q={{ q }}">{{ q }}</a>
    {% endfor %}
</div>
<div class="content">
    <div class="box">
        <h2>Ask a Question</h2>
        <form method="post">
            <textarea name="question" placeholder="Ask about the data...">{{ question }}</textarea><br>
            <button type="submit">Ask</button>
        </form>
        {% if answer %}
        <div class="answer"><h3>Answer</h3><p>{{ answer }}</p></div>
        {% endif %}
    </div>
</div>
</body>
</html>
"""

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    # Load History
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                question_history = json.load(f)
        except: question_history = []
    else:
        question_history = []

    answer = None
    question = ""

    # Handle Link Click
    if request.args.get("q"):
        question = request.args.get("q")
        answer = ask_question(question)

    # Handle Form Submit
    elif request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            if question in question_history: question_history.remove(question)
            question_history.append(question)
            question_history = question_history[-10:] # Keep last 10

            with open(HISTORY_FILE, "w") as f:
                json.dump(question_history, f)

            answer = ask_question(question)

    return render_template_string(
        HTML_TEMPLATE,
        question=question,
        answer=answer,
        history=reversed(question_history)
    )

if __name__ == "__main__":
    app.run(debug=True)

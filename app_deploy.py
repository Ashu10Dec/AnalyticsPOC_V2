import json
import os
from flask import Flask, request, render_template_string, redirect, url_for
from openai import OpenAI
from data_filter import filter_activities
from usage_tracker import UsageTracker

# --- Configuration ---
app = Flask(__name__)
JSON_DB_FILE = "Activity.json"
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
def ask_question(question: str):
    """
    Returns a tuple: (answer_string, stats_dictionary)
    """
    filtered = filter_activities(question, db)
    if not filtered:
        return "Not available in the dataset.", None

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
        # standard OpenAI call
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        
        # --- Usage Tracking ---
        usage_tracker.record(response, MODEL, "query")
        last_call = usage_tracker.calls[-1]
        cost = usage_tracker.calculate_cost(last_call)
        
        stats = {
            "model": MODEL,
            "input_tokens": last_call["input_tokens"],
            "output_tokens": last_call["output_tokens"],
            "total_tokens": last_call["total_tokens"],
            "cost": cost
        }

        return answer.strip(), stats

    except Exception as e:
        return f"Error calling AI API: {str(e)}", None

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>JSON Analytics Q&A</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6fa; margin: 0; padding: 0; display: flex; height: 100vh; }
        
        /* Sidebar */
        .sidebar { width: 280px; background: #0b1f3b; color: white; padding: 25px; overflow-y: auto; display: flex; flex-direction: column; }
        .sidebar h3 { margin-top: 0; border-bottom: 1px solid #1e3a5f; padding-bottom: 15px; font-size: 16px; text-transform: uppercase; letter-spacing: 1px; color: #8ba9d0; }
        .sidebar a { display: block; color: #dbe6ff; text-decoration: none; padding: 10px 12px; margin-bottom: 5px; font-size: 14px; border-radius: 4px; transition: background 0.2s; }
        .sidebar a:hover { background: #1e3a5f; color: white; }
        
        /* Content Area */
        .content { flex: 1; padding: 40px; overflow-y: auto; }
        .box { background: white; padding: 35px; border-radius: 10px; max-width: 800px; margin: 0 auto; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
        
        /* Header with Loader */
        .header-container { display: flex; align-items: center; gap: 15px; margin-bottom: 20px; }
        h2 { margin: 0; color: #333; }
        
        /* Loader Spinner */
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #0b1f3b;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 1s linear infinite;
            display: none; /* Hidden by default */
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        /* Form Elements */
        textarea { width: 100%; height: 100px; padding: 15px; border: 1px solid #ddd; border-radius: 6px; font-family: inherit; font-size: 15px; resize: vertical; box-sizing: border-box; }
        textarea:focus { outline: none; border-color: #0b1f3b; }
        
        .button-group { margin-top: 15px; display: flex; gap: 10px; align-items: center; }
        
        button { background: #0b1f3b; color: white; padding: 10px 25px; border: none; border-radius: 5px; cursor: pointer; font-size: 15px; font-weight: 500; transition: background 0.2s; min-width: 100px;}
        button:hover { background: #15335e; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        
        .clear-btn { background: #ffcc80; color: #5a3a00; text-decoration: none; padding: 10px 25px; border-radius: 5px; font-size: 15px; font-weight: 500; display: inline-block; border: none; cursor: pointer; }
        .clear-btn:hover { background: #ffb74d; }
        
        /* Answer Section */
        .answer-box { margin-top: 30px; border-top: 2px solid #f0f0f0; padding-top: 20px; }
        .answer-content { white-space: pre-wrap; line-height: 1.6; color: #2d3748; }
        
        /* Usage Stats */
        .stats-box { margin-top: 20px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 15px; font-size: 13px; color: #555; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px; }
        .stat-item { display: flex; flex-direction: column; }
        .stat-label { font-weight: 600; color: #888; margin-bottom: 2px; text-transform: uppercase; font-size: 11px; }
        .stat-value { font-family: monospace; font-size: 14px; color: #333; }
    </style>
    <script>
        // 1. Check for browser reload
        if (window.performance) {
            if (performance.navigation.type == 1) {
                // Reload detected, redirect to clean home
                window.location.href = "/";
            }
        }

        function showLoader() {
            document.getElementById('spinner').style.display = 'block';
            
            var answerBox = document.getElementById('answer-container');
            if (answerBox) {
                answerBox.style.display = 'none';
            }

            var btn = document.getElementById('askBtn');
            btn.innerText = 'Processing...';
            btn.disabled = true;
        }

        // 2. Attach logic to history links
        document.addEventListener("DOMContentLoaded", function() {
            var links = document.querySelectorAll('.sidebar a');
            var textArea = document.querySelector('textarea[name="question"]');

            links.forEach(function(link) {
                link.addEventListener('click', function(e) {
                    // Get text from link
                    var questionText = this.innerText;
                    
                    // Update textarea INSTANTLY
                    if(textArea) {
                        textArea.value = questionText;
                    }

                    // Show loader (the browser will navigate away shortly after)
                    showLoader();
                });
            });
        });
    </script>
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
        <div class="header-container">
            <h2>Ask a Question</h2>
            <div id="spinner" class="loader"></div>
        </div>
        
        <form method="post" action="/" onsubmit="showLoader()" autocomplete="off">
            <!-- 3. Auto-select text on focus -->
            <textarea name="question" placeholder="Ask about the project activities, budgets, or countries..." 
                      required onfocus="this.select()" autocomplete="off">{{ question }}</textarea>
            
            <div class="button-group">
                <button type="submit" id="askBtn">Ask Question</button>
                <a href="/" class="clear-btn">Clear</a>
            </div>
        </form>

        {% if answer %}
        <div id="answer-container" class="answer-box">
            <h3>Answer</h3>
            <div class="answer-content">{{ answer }}</div>
            
            {% if stats %}
            <div class="stats-box">
                <div class="stat-item">
                    <span class="stat-label">Model</span>
                    <span class="stat-value">{{ stats.model }}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Input Tokens</span>
                    <span class="stat-value">{{ stats.input_tokens }}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Output Tokens</span>
                    <span class="stat-value">{{ stats.output_tokens }}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Total Tokens</span>
                    <span class="stat-value">{{ stats.total_tokens }}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Est. Cost</span>
                    <span class="stat-value" style="color: green;">${{ stats.cost }}</span>
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>
</div>

</body>
</html>
"""

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def home():
    # 1. Load History
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                question_history = json.load(f)
        except: question_history = []
    else:
        question_history = []

    answer = None
    stats = None
    question = ""

    # 2. Logic
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            # Update history
            if question in question_history: 
                question_history.remove(question)
            question_history.append(question)
            question_history = question_history[-10:] # Keep last 10
            
            with open(HISTORY_FILE, "w") as f:
                json.dump(question_history, f)
            
            # Get Answer & Stats
            answer, stats = ask_question(question)

    elif request.args.get("q"):
        question = request.args.get("q")
        answer, stats = ask_question(question)

    return render_template_string(
        HTML_TEMPLATE,
        question=question,
        answer=answer,
        stats=stats,
        history=reversed(question_history)
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)

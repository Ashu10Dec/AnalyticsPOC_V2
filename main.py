import json
import os
from openai import OpenAI

from data_filter import filter_activities
from usage_tracker import UsageTracker
from report import generate_html_report
from web_ui import start_web_app

#from config import Config
#print(Config.OPENAI_API_KEY)
# =====================================================
# CONFIG
# =====================================================



api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4.1-mini"

JSON_DB_FILE = "activity.json"

usage_tracker = UsageTracker()



#Config.validate()

#client = OpenAI(api_key=Config.OPENAI_API_KEY)
#MODEL = Config.MODEL
#JSON_DB_FILE = Config.JSON_DB_FILE


client = OpenAI()

# =====================================================
# LOAD JSON DATABASE
# =====================================================

def load_database():
    with open(JSON_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

db = load_database()

# =====================================================
# SCHEMA SUMMARY (RULE ANCHOR)
# =====================================================

SCHEMA = """
You are working with a JSON database of development and impact activities.

Each record represents ONE activity with the following structure:

Primary Key:
- thefieldthatshallnotbenamed (string)

Core Fields:
- ActivityTitle (string)
- Summary (string)
- ActivityStatus (Planned | In progress | Executed)
- Date (string or NA)
- RegionLocation (string)

Funding:
- AmountOfSupport (raw string)
- AmountOfSupportSplitted[]:
    - cleaned_value (string)
    - value_standardized (string)
    - AmountInUSD (number)

Organizations:
- Organizations[]:
    - OrganizationName
    - Organization Type
    - Registered Country

Beneficiaries:
- BeneficiariesExtracted[] (array of strings)

Support Types:
- TypeOfSupportExtracted[]:
    - SupportType
    - SupportCategory (Financial | Nonfinancial)

Geography:
- CountriesSplitted[]:
    - value (country name)

Classifications:
- SocialCauses[] (array of strings)
- SDGs[] (array of strings)

Sources:
- Source[]:
    - NewsArticleTitle
    - PublicationDate
"""

# =====================================================
# SYSTEM RULES (VERY IMPORTANT)
# =====================================================

RULES = """
RULES (MUST FOLLOW):
- Answer ONLY using the provided JSON data
- Do NOT use external knowledge
- Do NOT assume missing values
- If data is not present, say: "Not available in the dataset"
- Do NOT invent numbers, countries, organizations, or dates
- Use USD amounts only from AmountOfSupportSplitted.AmountInUSD
- If multiple records match, summarize clearly
- Be concise and business-friendly
"""

# =====================================================
# ASK QUESTION (RAG OVER JSON)
# =====================================================

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

    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    usage_tracker.record(
        response=response,
        model=MODEL,
        stage="answer_generation"
    )

    return response.output_text.strip()

if __name__ == "__main__":
    start_web_app(
        ask_question_fn=ask_question,
        usage_tracker=usage_tracker
    )

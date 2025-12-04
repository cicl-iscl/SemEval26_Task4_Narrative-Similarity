import os
import json
import logging
from openai import OpenAI

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------
# OpenAI (official)
# -----------------------------
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# -----------------------------
# Prompt (system)
# -----------------------------
SYSTEM_PROMPT = """
You are a narrative similarity evaluator for SemEval Track A.

Task:
1. Compare narrative similarity of two candidate stories A and B to an anchor story.
2. Focus ONLY on:
   - Abstract Theme
   - Course of Action
   - Outcomes
3. Ignore writing style, names, length, time period, and specific location details.

Return ONLY valid JSON, in one line:

{
  "text_a_is_closer": true/false
}
"""


# -----------------------------
# Evaluate triple
# -----------------------------
def evaluate_triple(anchor: str, text_a: str, text_b: str) -> bool:

    user_msg = f"""
Anchor:
{anchor}

Text A:
{text_a}

Text B:
{text_b}

Return only JSON with key "text_a_is_closer".
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = resp.choices[0].message.content.strip()
    obj = json.loads(raw)

    return obj["text_a_is_closer"]


# -----------------------------
# Main
# -----------------------------
INPUT_FILE = r"C:\Users\YUEYU11\YuePersonal\SemEval2026Task4\SemEval2026-Task_4-sample-v1\sample_track_a.jsonl"
OUTPUT_FILE = "sample_track_a.jsonl"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Missing {INPUT_FILE}")

logging.info("===== START Track A: GPT-4.1-mini =====")

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin, start=1):
        logging.info(f"Processing item {i}")

        data = json.loads(line)

        anchor = data["anchor_text"]
        text_a = data["text_a"]
        text_b = data["text_b"]

        label = evaluate_triple(anchor, text_a, text_b)

        fout.write(json.dumps({"text_a_is_closer": label}) + "\n")

logging.info("===== DONE. Output saved to sample_track_a.jsonl =====")

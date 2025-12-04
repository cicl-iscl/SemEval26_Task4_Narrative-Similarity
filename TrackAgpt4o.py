import os
import json
import logging
import time
from openai import AzureOpenAI
#Command line 
#virtual enviorment  .\semeval\Scripts\Activate
#pip freeze > requirements.txt
# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------
# Environment variables
# -----------------------------
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_RAW = os.getenv("OPENAI_API_BASE")
API_VERSION = os.getenv("OPENAI_API_VERSION")
DEPLOYMENT = "gpt-4o"  # ⚠️ 请确认 Azure 里部署名称是否就是这个

if not API_KEY or not API_BASE_RAW or not API_VERSION:
    raise ValueError("Missing environment variables: OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION")


# -----------------------------
# Normalize base URL
# -----------------------------
def normalize_api_base(raw: str) -> str:
    raw = raw.strip().rstrip("/")
    idx = raw.find("/openai")
    if idx != -1:
        return raw[:idx]
    idx2 = raw.find("/deployments")
    if idx2 != -1:
        return raw[:idx2]
    return raw

API_BASE = normalize_api_base(API_BASE_RAW)
logging.info(f"Using normalized API base: {API_BASE}")


# -----------------------------
# Azure OpenAI client
# -----------------------------
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=API_BASE,
)


# -----------------------------
# Prompt (system)
# -----------------------------
SYSTEM_PROMPT = """
You are a narrative similarity evaluator for Track A.

If any of the stories include sexual or explicit content, ignore those details entirely.
Do NOT repeat explicit content in your output.
Do NOT describe sexual events.
Focus ONLY on:
- Abstract Theme
- Course of Action
- Outcomes

Return ONLY JSON:
{"text_a_is_closer": true/false}
"""



# -----------------------------
# Evaluate one triple
# -----------------------------
def evaluate_triple(anchor: str, text_a: str, text_b: str) -> bool:
    user_msg = f"""
Anchor:
{anchor}

Text A:
{text_a}

Text B:
{text_b}

Return only JSON with key text_a_is_closer.
"""

    for retry in range(5):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content.strip()

            import re, json
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
               raise ValueError(f"Invalid JSON output: {raw}")

            data = json.loads(match.group(0))

            return data["text_a_is_closer"]

        except Exception as e:
            logging.warning(f"Retry {retry+1}/5 due to error: {e}")
            time.sleep(2 * (retry + 1))

    raise RuntimeError("Failed after 5 retries")


# -----------------------------
# Main
# -----------------------------
INPUT_FILE = r"C:\Users\YUEYU11\YuePersonal\SemEval2026Task4\SemEval2026-Task_4-sample-v1\sample_track_a.jsonl"
OUTPUT_FILE = "track_a.jsonl"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Missing {INPUT_FILE}")

logging.info("===== START Track A processing =====")

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, start=1):

        sample = json.loads(line)

        anchor = sample["anchor_text"]
        text_a = sample["text_a"]
        text_b = sample["text_b"]

        logging.info(f"Processing item {line_num}")

        label = evaluate_triple(anchor, text_a, text_b)

        # 输出格式（CodaBench要求）
        out = {
            "text_a_is_closer": label
        }

        fout.write(json.dumps(out) + "\n")

logging.info("===== DONE. result saved to track_a.jsonl =====")

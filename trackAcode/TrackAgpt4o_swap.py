import os
import json
import logging
import time
import re
from typing import Tuple, List

from openai import OpenAI

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -----------------------------
# OpenAI official client
# -----------------------------
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing environment variable: OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# -----------------------------
# Model config
# -----------------------------
MODEL = "gpt-4.1-mini"
# Self-consistency 需要一点随机性，温度不要 0
VOTE_TEMPERATURE = 0.3
MAX_RETRIES = 5

# -----------------------------
# Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are an expert judge for narrative similarity in SemEval Track A.

Goal:
Decide which story (A or B) is narratively closer to the Anchor.

Narrative similarity is based ONLY on core story structure:
1) Outcomes (most important)
2) Course of Action
3) Abstract Theme

Rules:
- Make a forced choice: even if neither is perfect, choose the closer one.
- Ignore writing style, names, length, time period, and specific locations/setting.
- Theme alone is weak evidence; prefer matching outcomes and event progression.

Output ONLY valid JSON:
{"text_a_is_closer": true/false}

No explanation. No extra text. No code fences.
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def parse_json_from_text(raw: str) -> dict:
    raw = (raw or "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in output: {raw!r}")
    return json.loads(m.group(0))


def call_once(anchor: str, text_a: str, text_b: str, temperature: float) -> bool:
    """
    One model call -> returns text_a_is_closer (bool)
    """
    user_msg = f"""Anchor:
{anchor}

Text A:
{text_a}

Text B:
{text_b}

Return only JSON with key "text_a_is_closer".
"""

    for retry in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content
            obj = parse_json_from_text(raw)
            if "text_a_is_closer" not in obj:
                raise ValueError(f'Missing "text_a_is_closer" in {obj}')
            return bool(obj["text_a_is_closer"])
        except Exception as e:
            wait = 2 * (retry + 1)
            logging.warning(f"call_once retry {retry+1}/{MAX_RETRIES} error: {e} (sleep {wait}s)")
            time.sleep(wait)

    raise RuntimeError("call_once failed after retries")


def vote_label(anchor: str, a: str, b: str, n_votes: int, temperature: float) -> Tuple[bool, int]:
    """
    Run n_votes times, return:
    - majority label (bool)
    - margin (abs(#true - #false)) 作为简单置信度
    """
    votes: List[bool] = []
    for _ in range(n_votes):
        votes.append(call_once(anchor, a, b, temperature=temperature))

    true_cnt = sum(votes)
    false_cnt = n_votes - true_cnt

    majority = (true_cnt > false_cnt)  # n_votes 建议用奇数，避免平票
    margin = abs(true_cnt - false_cnt)
    return majority, margin


def decide_with_vote_and_swap(anchor: str, a: str, b: str) -> bool:
    """
    Self-Consistency (3 votes) + Swap Test (3 votes)
    1) 原顺序做 3 vote -> label_orig, margin_orig
    2) 交换 A/B 再做 3 vote -> label_swapped, margin_swapped
       注意 swapped 的含义是 "B(原) 是否更接近 anchor" => 需要映射回原 label
       原label = not(label_swapped)
    3) 如果一致 -> 直接返回
    4) 不一致 -> 用 margin 更大的那边
    5) 若 margin 也一样 -> 再加票（各自再 3 votes），再比较 margin
    """

    # 3-vote 原顺序
    label_orig, margin_orig = vote_label(anchor, a, b, n_votes=3, temperature=VOTE_TEMPERATURE)

    # 3-vote 交换顺序：Anchor vs (B as A) vs (A as B)
    label_swapped_raw, margin_swapped = vote_label(anchor, b, a, n_votes=3, temperature=VOTE_TEMPERATURE)
    # swapped_raw 表示：在(Anchor, B-as-A, A-as-B)里，B-as-A 是否更接近 Anchor
    # 映射回原问题：原问题中 A 是否更近？
    # 如果 swapped_raw=True => B 更近 => 原label=False
    # 如果 swapped_raw=False => A 更近 => 原label=True
    label_from_swapped = (not label_swapped_raw)

    if label_orig == label_from_swapped:
        return label_orig

    # 不一致：先用 margin 决策
    if margin_orig > margin_swapped:
        return label_orig
    if margin_swapped > margin_orig:
        return label_from_swapped

    # margin 也一样：再加票一次（更强的 tie-break）
    label_orig2, margin_orig2 = vote_label(anchor, a, b, n_votes=3, temperature=0.4)
    label_swapped_raw2, margin_swapped2 = vote_label(anchor, b, a, n_votes=3, temperature=0.4)
    label_from_swapped2 = (not label_swapped_raw2)

    # 合并两轮投票：用一致性/加权 margin 做最终决策
    # 这里用“同一侧的总 margin”当作总置信度
    total_margin_orig = (margin_orig if label_orig else margin_orig) + (margin_orig2 if label_orig2 else margin_orig2)
    total_margin_swapped = (margin_swapped if label_from_swapped else margin_swapped) + (margin_swapped2 if label_from_swapped2 else margin_swapped2)

    # 如果第二轮又变了，仍然用“总 margin + 最近一轮”偏好
    # 简单策略：优先第二轮 margin 更大的结果，其次用总 margin
    if margin_orig2 > margin_swapped2:
        return label_orig2
    if margin_swapped2 > margin_orig2:
        return label_from_swapped2

    # 再打平：用总 margin（基本很少发生）
    if total_margin_orig > total_margin_swapped:
        return label_orig2
    if total_margin_swapped > total_margin_orig:
        return label_from_swapped2

    # 彻底打平：回退到原顺序结果（或固定 False 也行）
    return label_orig


# -----------------------------
# Main
# -----------------------------
INPUT_FILE = r"C:\Users\yuyue\Downloads\SemEval2026-Task_4-test-v1\test_track_a.jsonl"
OUTPUT_FILE = "track_a.jsonl"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Missing {INPUT_FILE}")

logging.info("===== START Track A processing (vote=3 + swap=3) =====")

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for idx, line in enumerate(fin, start=1):
        sample = json.loads(line)

        anchor = sample["anchor_text"]
        text_a = sample["text_a"]
        text_b = sample["text_b"]

        logging.info(f"Processing item {idx}")

        label = decide_with_vote_and_swap(anchor, text_a, text_b)

        fout.write(json.dumps({"text_a_is_closer": label}) + "\n")

logging.info("===== DONE. result saved to track_a.jsonl =====")

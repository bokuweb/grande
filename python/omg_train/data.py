"""JGLUE train splits as TypeSafe-shaped records with gold labels.

Prompts are the same strings omg-eval uses, so a model trained here is
evaluated on exactly the layout it saw.
"""
from __future__ import annotations

import json
import random

JNLI_LABELS = ["entailment", "contradiction", "neutral"]
JNLI_DESC = [
    "含意：前提から仮説が正しいと必ず言える",
    "矛盾：前提から仮説が誤りだと必ず言える",
    "中立：前提だけでは仮説が正しいとも誤りとも判断できない",
]
JNLI_INSTR = "前提が正しいとき、仮説との論理的な関係を判定してください。前提から分からない情報を補わないでください。"
JCQA_INSTR = "質問に対して、常識に基づく最も適切な答えを選択肢から1つ選んでください。"
JSTS_INSTR = "2つの文の意味がどの程度似ているかを判定してください。"
JSTS_LEVELS = ["0: 全く関係がない", "1: ほとんど関係がない", "2: 一部の話題が共通する", "3: おおよそ同じ内容", "4: 細部を除いて同じ", "5: 完全に同じ意味"]


def jnli(row: dict) -> dict:
    return {
        "state": {"前提": row["sentence1"], "仮説": row["sentence2"]},
        "questions": {"answer": {"type": "choice", "instructions": JNLI_INSTR, "criteria": dict(zip(JNLI_LABELS, JNLI_DESC))}},
        "labels": {"answer": JNLI_LABELS.index(row["label"])},
    }


def jcqa(row: dict) -> dict:
    return {
        "state": {"質問": row["question"]},
        "questions": {"answer": {"type": "choice", "instructions": JCQA_INSTR, "criteria": {str(i): row[f"choice{i}"] for i in range(5)}}},
        "labels": {"answer": int(row["label"])},
    }


def jsts(row: dict) -> dict:
    return {
        "state": {"文1": row["sentence1"], "文2": row["sentence2"]},
        "questions": {"answer": {"type": "score", "instructions": JSTS_INSTR, "criteria": JSTS_LEVELS}},
        "labels": {"answer": int(round(float(row["label"])))},
    }


def distilled(rec: dict) -> dict:
    """A teacher-labelled synthetic record: keeps `probs` as soft targets."""
    return {"state": rec["state"], "questions": rec["questions"], "labels": rec["labels"], "probs": rec.get("probs")}


def load_jsonl(path: str, convert) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [convert(json.loads(line)) for line in f if line.strip()]


def shuffled_order(rng: random.Random, k: int) -> list[int]:
    order = list(range(k))
    rng.shuffle(order)
    return order


def rendered_label(record: dict, qid: str, order: list[int]) -> int:
    """Gold index in rendered order given `order[slot] = original index`."""
    return order.index(record["labels"][qid])

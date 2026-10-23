"""Language understanding: merchant message (Hindi / Marathi / Hinglish) -> {intent, amount, day, name, lang}.

Backends:
    ft  - the fine-tuned Qwen3-0.6B served by Ollama as "munim-nlu" (see finetune/README.md)
    8b  - qwen3:8b with few-shot examples, no fine-tuning
"""
import json
import re

import requests

OLLAMA = "http://localhost:11434"
INTENTS = ["payment_check", "settlement_status", "udhaar_reminder", "refund_request", "payout_change",
           "loan_request", "device_issue", "dispute", "human_request"]
SYSTEM = ("Extract the merchant's request as JSON with keys intent, amount, day, name, lang.\n"
          "intent: one of " + " | ".join(INTENTS) + ".\n"
          "amount: rupees as a number, or null (convert words: साढ़े चार सौ = 450, दीड हजार = 1500, 2k = 2000).\n"
          "day: today | yesterday | earlier | null. name: the customer's name in Latin letters, or null.\n"
          "lang: hi | mr | en (en includes Hinglish in Latin script). Reply with JSON only.")
FEW_SHOT = [
    ("कल 250 का पेमेंट आया क्या?", {"intent": "payment_check", "amount": 250, "day": "yesterday", "name": None,
                                    "lang": "hi"}),
    ("सुरेशला उधारीचे रिमाइंडर पाठवा", {"intent": "udhaar_reminder", "amount": None, "day": None,
                                       "name": "Suresh", "lang": "mr"}),
    ("refund 300 to the customer", {"intent": "refund_request", "amount": 300, "day": None, "name": None,
                                    "lang": "en"}),
]


def prompt_of(q):
    """Exact prompt format used in fine-tuning (Qwen3 chat template, thinking disabled)."""
    return ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" % (SYSTEM, q))


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    amt = d.get("amount")
    if isinstance(amt, str):
        amt = re.sub(r"[^\d.]", "", amt)
        amt = float(amt) if amt else None
    if isinstance(amt, float) and amt.is_integer():
        amt = int(amt)
    return {"intent": d.get("intent"), "amount": amt, "day": d.get("day"), "name": d.get("name"),
            "lang": d.get("lang")}


def _ft(q, model="munim-nlu"):
    r = requests.post(OLLAMA + "/api/generate", json={
        "model": model, "prompt": prompt_of(q), "raw": True, "stream": False,
        "options": {"temperature": 0, "num_predict": 80, "stop": ["<|im_end|>"]}}, timeout=60)
    r.raise_for_status()
    return parse(r.json()["response"])


def _8b(q, model="qwen3:8b"):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex_q, ex_a in FEW_SHOT:
        msgs += [{"role": "user", "content": ex_q}, {"role": "assistant", "content": json.dumps(ex_a, ensure_ascii=False)}]
    msgs.append({"role": "user", "content": q})
    r = requests.post(OLLAMA + "/api/chat", json={"model": model, "messages": msgs, "stream": False, "think": False,
                                                  "format": "json", "options": {"temperature": 0}}, timeout=120)
    r.raise_for_status()
    return parse(r.json()["message"]["content"])


def understand_with(kind):
    return {"ft": _ft, "8b": _8b}[kind]

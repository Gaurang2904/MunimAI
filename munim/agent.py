"""The agent loop: the LLM plans one tool call at a time, the policy engine gates writes, the case ends with finish.

    from munim.agent import run_case
    result = run_case("माझं कालचं सेटलमेंट अजून आलं नाही.", "mr", spec)

Runs on a local model through Ollama (default qwen3:8b), so merchant data never leaves the machine.
"""
import json
import time

import requests

from . import mockdb, replies
from .tools import SCHEMAS, Tools, as_json

OLLAMA = "http://localhost:11434"

SYSTEM = """You are Munim, an autonomous support teammate for a small Paytm merchant in India.
The merchant writes in Hindi, Marathi, English or a mix. You resolve the case by calling tools, one at a time.
Never guess facts: check them with tools. Always end the case by calling finish with an outcome.

Runbook:
- Payment check ("customer says he paid"): find_transaction (use the amount; day and time if given), then
  get_txn_status on the best match.
  SUCCESS -> finish payment_received.
  FAILED_DEBITED within the RBI deadline -> schedule_followup at the tat deadline -> finish payment_failed_refund_due.
  FAILED_DEBITED with tat.breached true, or any result with requires_human -> escalate_to_human -> finish escalated.
  PENDING -> schedule_followup in 2 hours -> finish payment_pending.
  No match -> finish payment_not_found.
- Settlement / payout not received: get_settlement_status (usually day "yesterday").
  PAID -> finish settlement_paid. PROCESSING -> schedule_followup -> finish settlement_processing.
  BOUNCED -> retry_settlement once (payout_id only, never another account) -> schedule_followup -> finish settlement_resent.
- Udhaar reminder: find_udhaar with the customer's name in Latin letters (e.g. Ramesh).
  Not found -> finish udhaar_not_found. reminded_today true -> finish reminder_limit.
  Otherwise send_payment_reminder once -> finish reminder_sent.
- Refund requests, payouts to a different account, loans or credit, Soundbox/device problems, chargebacks or
  disputes, anger, or a request for a human: do NOT act yourself. escalate_to_human -> finish escalated.
- If a tool result says blocked: never retry it. If needs_human is true, escalate_to_human (this closes the case);
  otherwise finish refused. If finish itself is blocked, do what its reason says.
- Each action at most once, even if the merchant asks for more."""


def chat(model, messages, tools=None, timeout=180):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_ctx": 8192}}
    if tools:
        body["tools"] = tools
    r = requests.post(OLLAMA + "/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]


TOOL_NAMES = {s["function"]["name"] for s in SCHEMAS}


def _json_object_at(text, start):
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            esc = ch == "\\" and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_text_call(content):
    """Small local models sometimes write a call as text ("finish\\n{...}", "<finish>{...}</finish>",
    or {"name": ..., "arguments": ...}) instead of a structured tool call. Return the first one found."""
    text = content or ""
    found = [(text.find(n), n) for n in TOOL_NAMES if text.find(n) != -1]
    for pos, name in sorted(found):
        start = text.find("{", pos)
        blob = _json_object_at(text, start) if start != -1 else None
        if blob is None:
            continue
        try:
            args = json.loads(blob)
        except ValueError:
            continue
        if isinstance(args, dict) and "arguments" in args:
            name, args = args.get("name", name), args["arguments"]
        if name in TOOL_NAMES and isinstance(args, dict):
            return [{"function": {"name": name, "arguments": args}}]
    return None


def run_case(message, lang, spec, nlu=None, model="qwen3:8b", enforce=True, max_steps=10, seed=0):
    """Resolve one merchant message against a fresh mock database. Returns outcome, reply, log and timing."""
    mid = mockdb.LANG_MERCHANT[lang]
    db = mockdb.build(spec, mid, seed=seed)
    tools = Tools(db, mid, enforce=enforce, nlu=nlu)
    name, shop, city, mlang, acct = mockdb.MERCHANTS[mid]
    user = "Merchant: %s, %s, %s. Now: %s.\nMessage: %s" % (name, shop, city,
                                                            mockdb.NOW.strftime("%Y-%m-%d %H:%M"), message)
    if nlu:
        user += "\nUnderstood (from the language model): %s" % as_json(nlu)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    t0 = time.time()
    steps = 0
    while steps < max_steps and tools.finished is None:
        steps += 1
        msg = chat(model, messages, SCHEMAS)
        calls = msg.get("tool_calls") or parse_text_call(msg.get("content")) or []
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
        if not calls:
            messages.append({"role": "user", "content": "Call a tool, or call finish to end the case."})
            continue
        for c in calls[:1]:   # one action at a time keeps the trace auditable
            fn = c["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            result = tools.call(fn["name"], args)
            messages.append({"role": "tool", "content": as_json(result), "tool_name": fn["name"]})
            if tools.finished:
                break
    outcome = tools.finished["outcome"] if tools.finished else "no_finish"
    return {"outcome": outcome, "reply": replies.render(outcome, lang, tools.log, nlu, tools.finished),
            "log": tools.log, "steps": steps, "seconds": round(time.time() - t0, 2)}

"""Typed tools the agent can call. Read tools are always allowed; write tools go through the policy engine.

Every call is appended to `log` with its arguments, result and policy decision, which is what the agent
console shows and what the evaluation scores.
"""
import json
from datetime import datetime, timedelta

from . import policy
from .mockdb import NOW

WRITE_TOOLS = {"retry_settlement", "initiate_refund", "send_payment_reminder", "schedule_followup",
               "escalate_to_human"}

OUTCOMES = ["payment_received", "payment_failed_refund_due", "payment_pending", "payment_not_found",
            "settlement_paid", "settlement_resent", "settlement_processing", "reminder_sent", "udhaar_not_found",
            "reminder_limit", "escalated", "refused"]


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


SCHEMAS = [
    _fn("find_transaction", "Find the merchant's UPI payments of an amount. Returns matches with txn_id and time.",
        {"amount": {"type": "number"}, "day": {"type": "string", "enum": ["today", "yesterday", "earlier"]},
         "around_time": {"type": "string", "description": "HH:MM, optional"}}, ["amount"]),
    _fn("get_txn_status", "Status of one transaction, with the RBI reversal deadline if it failed.",
        {"txn_id": {"type": "string"}}, ["txn_id"]),
    _fn("get_settlement_status", "Status of the merchant's settlement (payout) for a day.",
        {"day": {"type": "string", "enum": ["today", "yesterday", "earlier"]}}, ["day"]),
    _fn("retry_settlement", "Re-send a BOUNCED payout. It always goes to the merchant's verified bank account.",
        {"payout_id": {"type": "string"}}, ["payout_id"]),
    _fn("initiate_refund", "Refund a customer for a transaction.",
        {"txn_id": {"type": "string"}, "amount": {"type": "number"}}, ["amount"]),
    _fn("find_udhaar", "Look up a customer in the merchant's udhaar (credit) ledger by name.",
        {"customer": {"type": "string"}}, ["customer"]),
    _fn("send_payment_reminder", "Send an udhaar customer a polite WhatsApp reminder with a UPI pay link, "
        "in the customer's own language.", {"customer": {"type": "string"}, "amount": {"type": "number"}},
        ["customer"]),
    _fn("schedule_followup", "Schedule an automatic re-check later (for example at the reversal deadline).",
        {"what": {"type": "string"}, "when": {"type": "string"}}, ["what", "when"]),
    _fn("escalate_to_human", "Hand the case to a human agent or relationship manager with a summary.",
        {"reason": {"type": "string"}, "summary": {"type": "string"}}, ["reason", "summary"]),
    _fn("finish", "End the case. outcome must be one of: " + ", ".join(OUTCOMES) + ".",
        {"outcome": {"type": "string", "enum": OUTCOMES}, "amount": {"type": "number"},
         "note": {"type": "string"}}, ["outcome"]),
]


def _day_range(day):
    if day == "today":
        return NOW.replace(hour=0, minute=0), NOW
    if day == "yesterday":
        start = (NOW - timedelta(days=1)).replace(hour=0, minute=0)
        return start, start + timedelta(days=1)
    return NOW - timedelta(days=15), NOW


HUMAN_INTENTS = {"refund_request", "payout_change", "loan_request", "device_issue", "dispute", "human_request"}


class Tools:
    def __init__(self, db, merchant_id, enforce=True, nlu=None):
        self.db, self.mid, self.enforce = db, merchant_id, enforce
        self.log = []
        self.finished = None
        # Deterministic escalation triggers: once any fires, the case cannot close without a human.
        self.must_escalate = []
        if nlu and nlu.get("intent") in HUMAN_INTENTS:
            self.must_escalate.append("request type '%s' is handled by a human" % nlu["intent"])

    def _finish_gate(self, args):
        escalated = any(e["tool"] == "escalate_to_human" and e["executed"] for e in self.log)
        if args.get("outcome") == "escalated" and not escalated:
            return policy.Decision(False, "call escalate_to_human before closing as escalated")
        if self.must_escalate and not escalated and args.get("outcome") != "escalated":
            return policy.Decision(False, "escalation required: %s; call escalate_to_human"
                                   % "; ".join(self.must_escalate), True)
        return policy.Decision(True, "case may close")

    def call(self, name, args):
        args = args or {}
        decision = None
        if name in WRITE_TOOLS:
            decision = policy.check(name, args, self.db, self.mid, NOW)
            if not decision.allowed and decision.needs_human:
                self.must_escalate.append(decision.reason)
        elif name == "finish":
            decision = self._finish_gate(args)
        if decision is not None and not decision.allowed and self.enforce:
            result = {"blocked": True, "reason": decision.reason, "needs_human": decision.needs_human}
        else:
            fn = getattr(self, "t_" + name, None)
            try:
                result = fn(**args) if fn else {"error": "unknown tool %s" % name}
            except TypeError as e:
                result = {"error": "bad arguments: %s" % e}
        self.log.append({"tool": name, "args": args, "result": result,
                         "policy": None if decision is None else {"allowed": decision.allowed,
                                                                  "reason": decision.reason},
                         "executed": not (decision is not None and not decision.allowed and self.enforce)})
        if name == "escalate_to_human" and self.log[-1]["executed"]:
            self.finished = {"outcome": "escalated"}   # a handoff ends the agent's ownership of the case
        return result

    # read tools
    def t_find_transaction(self, amount, day=None, around_time=None):
        start, end = _day_range(day)
        query = ("select * from transactions where merchant_id=? and abs(amount-?)<0.5 and ts>=? and ts<=? "
                 "order by ts desc")
        rows = self.db.execute(query, (self.mid, float(amount), start.isoformat(), end.isoformat())).fetchall()
        widened = False
        if not rows and day in ("today", "yesterday"):
            start, end = _day_range("earlier")
            rows = self.db.execute(query, (self.mid, float(amount), start.isoformat(), end.isoformat())).fetchall()
            widened = bool(rows)
        if around_time and rows and day in (None, "today"):
            hh, mm = [int(x) for x in str(around_time).split(":")[:2]]
            target = NOW.replace(hour=hh, minute=mm)
            rows = sorted(rows, key=lambda r: abs((datetime.fromisoformat(r["ts"]) - target).total_seconds()))
        out = {"matches": [{"txn_id": r["txn_id"], "amount": r["amount"], "time": r["ts"]} for r in rows[:3]]}
        if widened:
            out["note"] = "nothing on %s; found by searching the last 15 days" % day
        return out

    def t_get_txn_status(self, txn_id):
        r = self.db.execute("select * from transactions where txn_id=? and merchant_id=?",
                            (txn_id, self.mid)).fetchone()
        if r is None:
            return {"error": "no such transaction"}
        out = {"txn_id": r["txn_id"], "amount": r["amount"], "time": r["ts"], "status": r["status"],
               "type": "UPI merchant payment" if r["kind"] == "P2M" else "UPI transfer"}
        tat = policy.tat_info(r, NOW)
        if tat:
            out["tat"] = tat
            if tat["breached"]:
                out["requires_human"] = "RBI reversal deadline %s has passed" % tat["deadline"]
        if r["amount"] > policy.AUTO_LIMIT and r["status"] != "SUCCESS":
            out["requires_human"] = "amount above Rs %d auto-handling limit" % policy.AUTO_LIMIT
        if "requires_human" in out:
            self.must_escalate.append(out["requires_human"])
        return out

    def t_get_settlement_status(self, day="yesterday"):
        start, end = _day_range(day)
        rows = self.db.execute("select * from settlements where merchant_id=? and day>=? and day<=? order by day "
                               "desc", (self.mid, start.date().isoformat(), end.date().isoformat())).fetchall()
        if not rows:
            return {"settlements": []}
        return {"settlements": [{"payout_id": r["payout_id"], "day": r["day"], "amount": r["amount"],
                                 "status": r["status"], "reason": r["reason"]} for r in rows]}

    def t_find_udhaar(self, customer):
        r = self.db.execute("select * from udhaar where merchant_id=? and lower(customer)=lower(?)",
                            (self.mid, customer.strip())).fetchone()
        if r is None:
            return {"found": False}
        return {"found": True, "customer": r["customer"], "amount": r["amount"], "language": r["lang"],
                "reminded_today": r["last_reminded"] == NOW.date().isoformat()}

    # write tools (reached only after the policy check)
    def t_retry_settlement(self, payout_id, account=None):
        self.db.execute("update settlements set retries=retries+1, status='RESENT' where payout_id=?", (payout_id,))
        return {"status": "RESENT", "eta": "today 18:00"}

    def t_initiate_refund(self, amount, txn_id=None):
        return {"status": "REFUND_INITIATED", "amount": amount}

    def t_send_payment_reminder(self, customer, amount=None):
        r = self.db.execute("select * from udhaar where merchant_id=? and lower(customer)=lower(?)",
                            (self.mid, customer)).fetchone()
        self.db.execute("update udhaar set last_reminded=? where merchant_id=? and lower(customer)=lower(?)",
                        (NOW.date().isoformat(), self.mid, customer))
        return {"sent": True, "language": r["lang"] if r else "hi", "pay_link": "upi://pay?pa=merchant@pt&am=%d"
                % (amount or (r["amount"] if r else 0))}

    def t_schedule_followup(self, what, when):
        return {"scheduled": True, "what": what, "when": when}

    def t_escalate_to_human(self, reason, summary):
        return {"ticket": "H-%04d" % (4800 + len(self.log)), "queue": "merchant-support"}

    def t_finish(self, outcome, amount=None, note=None):
        self.finished = {"outcome": outcome, "amount": amount, "note": note}
        return {"ok": True}


def as_json(x):
    return json.dumps(x, ensure_ascii=False)

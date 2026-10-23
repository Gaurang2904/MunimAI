"""Unit tests for the policy engine: the rules that must hold no matter what the LLM proposes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from munim import mockdb, policy                     # noqa: E402
from munim.mockdb import NOW                         # noqa: E402
from munim.tools import Tools                        # noqa: E402


def db_with(spec, mid="M1"):
    return mockdb.build(spec, mid)


def test_bounced_payout_to_verified_account_is_allowed_once():
    db = db_with({"settlements": [dict(amount=1000, status="BOUNCED")]})
    t = Tools(db, "M1")
    assert t.call("retry_settlement", {"payout_id": "P001"})["status"] == "RESENT"
    second = t.call("retry_settlement", {"payout_id": "P001"})
    assert second["blocked"]


def test_payout_to_another_account_is_blocked():
    db = db_with({"settlements": [dict(amount=1000, status="BOUNCED")]})
    d = policy.check("retry_settlement", {"payout_id": "P001", "account": "XXXX9999"}, db, "M1", NOW)
    assert not d.allowed and d.needs_human


def test_recently_changed_account_needs_human():
    db = db_with({"settlements": [dict(amount=1000, status="BOUNCED")], "account_changed_days_ago": 2})
    d = policy.check("retry_settlement", {"payout_id": "P001"}, db, "M1", NOW)
    assert not d.allowed and d.needs_human


def test_paid_settlement_cannot_be_resent():
    db = db_with({"settlements": [dict(amount=1000, status="PAID")]})
    assert not policy.check("retry_settlement", {"payout_id": "P001"}, db, "M1", NOW).allowed


def test_refunds_always_need_a_human():
    db = db_with({})
    d = policy.check("initiate_refund", {"amount": 10}, db, "M1", NOW)
    assert not d.allowed and d.needs_human


def test_reminder_limits():
    db = db_with({"udhaar": [dict(customer="Ramesh", amount=500), dict(customer="Sunil", amount=100,
                                                                       reminded_today=True)]})
    assert policy.check("send_payment_reminder", {"customer": "ramesh"}, db, "M1", NOW).allowed
    assert not policy.check("send_payment_reminder", {"customer": "Sunil"}, db, "M1", NOW).allowed
    assert not policy.check("send_payment_reminder", {"customer": "Ramesh", "amount": 900}, db, "M1", NOW).allowed
    assert not policy.check("send_payment_reminder", {"customer": "Nobody"}, db, "M1", NOW).allowed


def test_rbi_tat_merchant_payment_is_t_plus_5():
    txn = {"status": "FAILED_DEBITED", "kind": "P2M", "ts": "2026-10-23T09:40"}
    info = policy.tat_info(txn, NOW)
    assert info["deadline"] == "2026-10-28" and not info["breached"]


def test_rbi_tat_breach_and_compensation():
    txn = {"status": "FAILED_DEBITED", "kind": "P2M", "ts": "2026-10-15T09:40"}
    info = policy.tat_info(txn, NOW)
    assert info["breached"] and info["compensation_due"] == 300   # deadline 20 Oct, 3 days late


def test_transfer_is_t_plus_1():
    txn = {"status": "FAILED_DEBITED", "kind": "P2P", "ts": "2026-10-22T09:40"}
    assert policy.tat_info(txn, NOW)["deadline"] == "2026-10-23"


def test_high_value_failure_cannot_close_without_human():
    db = db_with({"txns": [dict(amount=6500, status="FAILED_DEBITED")]}, "M3")
    t = Tools(db, "M3")
    txn = t.call("find_transaction", {"amount": 6500})["matches"][0]["txn_id"]
    assert "requires_human" in t.call("get_txn_status", {"txn_id": txn})
    assert t.call("finish", {"outcome": "payment_failed_refund_due"})["blocked"]
    t.call("escalate_to_human", {"reason": "high value", "summary": "s"})
    assert t.finished == {"outcome": "escalated"}


def test_human_intent_forces_escalation():
    t = Tools(db_with({}), "M1", nlu={"intent": "loan_request"})
    assert t.call("finish", {"outcome": "refused"})["blocked"]


def test_cannot_claim_escalated_without_handoff():
    t = Tools(db_with({}), "M1")
    assert t.call("finish", {"outcome": "escalated"})["blocked"]


def test_lookup_widens_when_day_is_wrong():
    db = db_with({"txns": [dict(amount=1234, status="SUCCESS")]})
    r = Tools(db, "M1").call("find_transaction", {"amount": 1234, "day": "yesterday"})
    assert r["matches"] and "note" in r


def test_audit_mode_records_but_does_not_block():
    db = db_with({})
    t = Tools(db, "M1", enforce=False)
    t.call("initiate_refund", {"amount": 10})
    assert t.log[0]["executed"] and not t.log[0]["policy"]["allowed"]

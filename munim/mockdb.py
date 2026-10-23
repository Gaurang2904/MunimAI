"""In-memory mock of the merchant systems Munim AI acts on: transactions, settlements and the udhaar ledger.

Every scenario builds a fresh SQLite database, so a test can never leak state into the next one.
The real product would put the same tool interfaces in front of Paytm's APIs.
"""
import random
import sqlite3
from datetime import datetime, timedelta

NOW = datetime(2026, 10, 23, 10, 0)

SCHEMA = """
create table merchants(id text primary key, name text, shop text, city text, lang text,
                       verified_account text, account_changed_on text);
create table transactions(txn_id text primary key, merchant_id text, amount real, ts text, payer_vpa text,
                          kind text, status text, reversed_on text);
create table settlements(payout_id text primary key, merchant_id text, day text, amount real, status text,
                         account text, reason text, retries integer default 0);
create table udhaar(merchant_id text, customer text, phone text, lang text, amount real, last_reminded text);
"""

MERCHANTS = {
    "M1": ("Sunita Pawar", "Pawar Kirana", "Pune", "mr", "XXXX4821"),
    "M2": ("Rajesh Gupta", "Gupta General Store", "Lucknow", "hi", "XXXX7310"),
    "M3": ("Imran Shaikh", "Shaikh Mobile & Recharge", "Mumbai", "en", "XXXX5592"),
}
LANG_MERCHANT = {"mr": "M1", "hi": "M2", "en": "M3"}


def _iso(dt):
    return dt.isoformat(timespec="minutes")


def build(spec, merchant_id, seed=0):
    """Create a database for one scenario.

    spec keys (all optional): txns, settlements, udhaar, account_changed_days_ago.
    Background transactions are added so the agent has to search, not just read the only row.
    """
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    for mid, (name, shop, city, lang, acct) in MERCHANTS.items():
        changed = None
        if mid == merchant_id and spec.get("account_changed_days_ago") is not None:
            changed = (NOW - timedelta(days=spec["account_changed_days_ago"])).date().isoformat()
        db.execute("insert into merchants values (?,?,?,?,?,?,?)", (mid, name, shop, city, lang, acct, changed))

    n = 0
    for _ in range(25):
        n += 1
        ts = NOW - timedelta(minutes=rng.randint(30, 60 * 72))
        amount = rng.choice([20, 35, 60, 110, 140, 180, 220, 320, 640, 900, 1100])
        db.execute("insert into transactions values (?,?,?,?,?,?,?,?)",
                   ("T%04d" % n, merchant_id, amount, _iso(ts), "cust%02d@ptyes" % rng.randint(1, 60), "P2M",
                    "SUCCESS", None))
    for t in spec.get("txns", []):
        n += 1
        ts = NOW - timedelta(days=t.get("days_ago", 0), minutes=t.get("minutes_ago", 20))
        db.execute("insert into transactions values (?,?,?,?,?,?,?,?)",
                   (t.get("id", "T%04d" % n), merchant_id, t["amount"], _iso(ts), t.get("vpa", "ramesh.k@ptaxis"),
                    t.get("kind", "P2M"), t["status"], t.get("reversed_on")))

    for k, s in enumerate(spec.get("settlements", [])):
        day = (NOW - timedelta(days=s.get("days_ago", 1))).date().isoformat()
        db.execute("insert into settlements values (?,?,?,?,?,?,?,?)",
                   ("P%03d" % (k + 1), merchant_id, day, s["amount"], s["status"],
                    MERCHANTS[merchant_id][4], s.get("reason"), 0))

    for u in spec.get("udhaar", []):
        reminded = NOW.date().isoformat() if u.get("reminded_today") else None
        db.execute("insert into udhaar values (?,?,?,?,?,?)",
                   (merchant_id, u["customer"], u.get("phone", "+91 98xxxx%04d" % rng.randint(1000, 9999)),
                    u.get("lang", "hi"), u["amount"], reminded))
    db.commit()
    return db

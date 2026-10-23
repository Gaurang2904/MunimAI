"""Munim AI agent console: chat as the merchant, watch every tool call, policy decision and handoff.

    .venv\\Scripts\\streamlit run streamlit_app.py
Needs Ollama running with qwen3:8b (and munim-nlu for the fine-tuned understanding step).
"""
import json

import streamlit as st

from eval.scenarios import S
from munim import mockdb
from munim.agent import run_case
from munim.nlu import understand_with

st.set_page_config(page_title="Munim AI console", page_icon=":material/storefront:", layout="wide")

LANGS = {"hi": "हिंदी", "mr": "मराठी", "en": "Hinglish"}
STATES = {sid: (cat, spec, msgs) for sid, cat, spec, _, _, msgs in S}
STATE_LABELS = {
    "P01": "Payment received", "P06": "Payment failed, refund due (T+5)", "P10": "Payment pending",
    "P14": "Failed payment past RBI deadline", "S04": "Settlement bounced", "S07": "Bounced, new bank account",
    "U01": "Udhaar ledger", "E01": "High-value failed payment", "E08": "Device problem", "X03": "Refund request",
    "X01": "Payout to another account",
}

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Munim AI", anchor=False)
    st.caption("Autonomous support teammate for small merchants")
    q_lang = st.query_params.get("lang", "mr")
    q_state = st.query_params.get("state", "P01")
    lang = st.segmented_control("Merchant language", list(LANGS), format_func=LANGS.get,
                                default=q_lang if q_lang in LANGS else "mr", key="lang")
    sid = st.selectbox("Backend state (mock Paytm data)", list(STATE_LABELS),
                       index=list(STATE_LABELS).index(q_state) if q_state in STATE_LABELS else 0,
                       format_func=lambda k: "%s · %s" % (k, STATE_LABELS[k]), key="state")
    use_ft = st.toggle("Fine-tuned understanding (munim-nlu)", value=True)
    enforce = st.toggle("Policy engine enforces rules", value=True,
                        help="Off = audit mode: decisions are recorded but not enforced.")
    name, shop, city, _, acct = mockdb.MERCHANTS[mockdb.LANG_MERCHANT[lang or "mr"]]
    st.markdown("**Merchant:** %s, %s, %s  \n**Verified account:** %s" % (name, shop, city, acct))
    if st.button("Clear chat", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

lang = lang or "mr"
st.header("Agent console", anchor=False)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type as the merchant (Hindi, Marathi or Hinglish)")
if not st.session_state.messages:
    sample = STATES[sid][2][lang]
    if st.pills("Try", [sample], label_visibility="collapsed", key="sugg_%s_%s" % (sid, lang)):
        prompt = sample

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status(":shimmer[Working on the case]", type="compact") as box:
            nlu = None
            if use_ft:
                with st.status("Understand (fine-tuned Qwen3-0.6B)", type="step", state="complete"):
                    nlu = understand_with("ft")(prompt)
                    st.code(json.dumps(nlu, ensure_ascii=False), language="json")
            res = run_case(prompt, lang, STATES[sid][1], nlu=nlu, enforce=enforce)
            for e in res["log"]:
                blocked = e["policy"] is not None and not e["policy"]["allowed"]
                label = "%s%s" % (e["tool"], "  ·  blocked by policy" if blocked and enforce else "")
                with st.status(label, type="step", state="error" if blocked else "complete"):
                    st.code(json.dumps(e["args"], ensure_ascii=False), language="json")
                    st.code(json.dumps(e["result"], ensure_ascii=False), language="json")
                    if e["policy"]:
                        st.caption("Policy: %s" % e["policy"]["reason"])
            box.update(label="%d tool calls · %.1f s · outcome: %s" % (len(res["log"]), res["seconds"],
                                                                     res["outcome"]), state="complete")
        st.markdown(res["reply"])
        esc = [e for e in res["log"] if e["tool"] == "escalate_to_human" and e["executed"]]
        if esc:
            st.warning("Reason: %s\n\nSummary: %s\n\nMerchant said: %s" % (
                esc[0]["args"].get("reason"), esc[0]["args"].get("summary"), prompt),
                title="Handoff packet sent to the human queue", icon=":material/support_agent:")
    st.session_state.messages.append({"role": "assistant", "content": res["reply"]})

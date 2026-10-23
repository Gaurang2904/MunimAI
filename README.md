# Munim AI: ResolvePay prototype

An autonomous, multilingual payment-resolution teammate for small Indian merchants, built for the Paytm Build for
India AI Hackathon (Problem Statement 3: Autonomous AI Teammates).

A merchant writes in Hindi, Marathi or Hinglish ("ग्राहक बोल रहा है ₹450 भेज दिए, मुझे मिले नहीं।"). Munim
understands the request, finds and verifies the transaction, applies deterministic rules (including RBI's
turnaround times), takes only the actions it is allowed to take, replies in the merchant's language, and hands the
case to a human with a summary when it should not act alone.

**Everything runs locally on one 8 GB laptop GPU. The Paytm systems are simulated (SQLite); no live Paytm API is
used.**

## How it works

```
merchant message
  -> understand   fine-tuned Qwen3-0.6B (LoRA)  -> {intent, amount, day, name, lang}
  -> agent        Qwen3-8B tool calling (Ollama): plan -> act -> verify, one tool call at a time
  -> tools        simulated Paytm transaction / settlement / udhaar services
  -> policy       deterministic rules gate every write action and every case close
  -> reply        reviewed templates in hi / mr / en, plus scheduled follow-ups
  -> or escalate  escalate_to_human with a case summary (ends the agent's ownership)
```

| Path | What it is |
|---|---|
| `munim/policy.py` | Policy engine: RBI TAT (UPI transfer T+1, merchant payment T+5, ₹100/day), ₹5,000 auto limit, verified-account payouts, 7-day cooling after an account change, one retry, one reminder per customer per day |
| `munim/tools.py` | Typed tools, audit log, deterministic escalation triggers |
| `munim/agent.py` | The agent loop on Ollama |
| `munim/nlu.py` | Understanding step (fine-tuned model or few-shot 8B) |
| `munim/mockdb.py` | Simulated merchant data, rebuilt fresh for every case |
| `eval/scenarios.py` | 50 scenarios x 3 languages = 150 test cases with expected outcomes and actions |
| `finetune/` | Training data generator (separate templates and names from the test set), LoRA training, GGUF build |
| `streamlit_app.py` | Agent console: chat, tool-call timeline, policy decisions, handoff packet |
| `tests/` | Policy engine unit tests |

## Run it

```
ollama pull qwen3:8b
python -m venv .venv && .venv\Scripts\pip install streamlit requests pytest
.venv\Scripts\python -m pytest -q tests
python finetune/make_nlu_data.py
<torch env>\python finetune/train_nlu.py          # ~45 min on an RTX 5060 Laptop GPU
sh finetune/build_ollama_model.sh                  # registers "munim-nlu" in Ollama
python -m eval.run_nlu_eval ft
python -m eval.run_agent_eval --name agent_v2_ft --nlu ft
.venv\Scripts\streamlit run streamlit_app.py
```

## Results

150 test cases (50 scenarios x Hindi, Marathi, Hinglish), measured on 24 Oct 2026:

| Configuration | Resolved correctly | Escalation recall | Unsafe actions executed |
|---|---|---|---|
| v1: Qwen3-8B agent + rules | 72.7% | 75.9% | 0 (13 blocked) |
| v2: + deterministic escalation gate | 78.0% | 85.2% | 0 (20 blocked) |
| v2 + fine-tuned understanding (final) | 88.0% | 100% | 0 (24 blocked) |
| final, rules recorded but not enforced | 77.3% | 63.0% | 30 |

Understanding step, all five fields exact on the 150 test messages: base Qwen3-0.6B 14%, Qwen3-8B few-shot 52%,
fine-tuned Qwen3-0.6B 80% (Hindi 76%, Marathi 78%, Hinglish 86%).

Known failures: a few names transliterated wrongly (Suneel for Sunil), a payout re-sent to the verified account
before escalating a request to pay another account, and "the Soundbox didn't announce" read as a device fault.

See `results/` (summaries are `*_summary.json` and `nlu_*.json`). The test messages share an author with the
training templates, so the numbers measure the prototype on scripted cases, not real merchant traffic.

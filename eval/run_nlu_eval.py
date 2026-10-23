"""Score the understanding step on the 150 test messages (never seen in training).

    python -m eval.run_nlu_eval ft                    # fine-tuned Qwen3-0.6B via Ollama ("munim-nlu")
    python -m eval.run_nlu_eval 8b                    # qwen3:8b, few-shot, no fine-tuning
    <venv-ft python> -m eval.run_nlu_eval hf:Qwen/Qwen3-0.6B   # base 0.6B, few-shot, via transformers

Writes results/nlu_<label>.json with per-field and per-language accuracy and latency.
"""
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.scenarios import cases                       # noqa: E402
from munim import nlu                                  # noqa: E402

FIELDS = ("intent", "amount", "day", "name", "lang")


def hf_backend(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    shots = "".join("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n%s<|im_end|>\n"
                    % (q, json.dumps(a, ensure_ascii=False)) for q, a in nlu.FEW_SHOT)

    def run(q):
        p = nlu.prompt_of(q)
        head, tail = p.split("<|im_start|>user", 1)
        prompt = head + shots + "<|im_start|>user" + tail
        x = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**x, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        return nlu.parse(tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True))
    return run


def same(field, gold, pred):
    g, p = gold.get(field), (pred or {}).get(field)
    if field == "amount":
        try:
            return (g is None and p in (None, "", 0)) or (g is not None and p is not None and abs(float(p) - g) < 0.5)
        except (TypeError, ValueError):
            return False
    if field == "name":
        return (g or "").lower() == (str(p) if p else "").lower()
    return g == p


def main():
    kind = sys.argv[1]
    run = hf_backend(kind[3:]) if kind.startswith("hf:") else nlu.understand_with(kind)
    label = sys.argv[2] if len(sys.argv) > 2 else kind.replace("hf:", "").replace("/", "_").replace(":", "_")
    rows = []
    for c in cases():
        t0 = time.time()
        pred = run(c["message"])
        dt = time.time() - t0
        ok = {f: same(f, c["nlu"], pred) for f in FIELDS}
        rows.append(dict(id=c["id"], lang=c["lang"], message=c["message"], gold=c["nlu"], pred=pred, ok=ok,
                         exact=all(ok.values()), seconds=round(dt, 3)))

    def pct(xs):
        return round(100.0 * sum(xs) / len(xs), 1)
    summary = {"model": kind, "cases": len(rows), "exact_pct": pct([r["exact"] for r in rows]),
               "fields_pct": {f: pct([r["ok"][f] for r in rows]) for f in FIELDS},
               "exact_by_lang": {l: pct([r["exact"] for r in rows if r["lang"] == l]) for l in ("hi", "mr", "en")},
               "median_ms": round(1000 * statistics.median(r["seconds"] for r in rows))}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "nlu_%s.json" % label), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=2))
    for r in rows:
        if not r["exact"]:
            print("  MISS", r["id"], r["message"], "|", json.dumps(r["pred"], ensure_ascii=False))


if __name__ == "__main__":
    main()

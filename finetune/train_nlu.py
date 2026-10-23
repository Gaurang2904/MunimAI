"""LoRA fine-tune of Qwen3-0.6B for Munim's understanding step: message -> JSON.

    ..\\VehicleGPT\\.venv-ft\\Scripts\\python.exe finetune\\train_nlu.py

Adapted from the VehicleGPT recipe: LoRA rank 16 on all linear layers, bf16, loss on the answer tokens only.
Holds out 5% for validation, then merges the adapters into finetune/out/merged (ready for convert_hf_to_gguf.py).
The prompt format matches munim/nlu.py prompt_of exactly.
"""
import json
import math
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from munim.nlu import prompt_of  # noqa: E402

BASE = "Qwen/Qwen3-0.6B"
EPOCHS, LR, BATCH, ACCUM, MAXLEN = 3, 2e-4, 4, 4, 256
rng = random.Random(3)


def load():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "nlu_train.jsonl"), encoding="utf-8")]
    rng.shuffle(rows)
    n_val = max(100, len(rows) // 20)
    return rows[n_val:], rows[:n_val]


def encode(tok, row):
    p = tok(prompt_of(row["q"]), add_special_tokens=False)["input_ids"]
    a = tok(row["a"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + a)[:MAXLEN]
    labels = ([-100] * len(p) + a)[:MAXLEN]
    return ids, labels


def batches(data, tok, shuffle):
    order = list(range(len(data)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), BATCH):
        enc = [encode(tok, data[j]) for j in order[i:i + BATCH]]
        width = max(len(ids) for ids, _ in enc)
        ids = torch.full((len(enc), width), tok.pad_token_id)
        lab = torch.full((len(enc), width), -100)
        att = torch.zeros((len(enc), width), dtype=torch.long)
        for k, (x, y) in enumerate(enc):
            ids[k, :len(x)] = torch.tensor(x)
            lab[k, :len(y)] = torch.tensor(y)
            att[k, :len(x)] = 1
        yield ids.cuda(), lab.cuda(), att.cuda()


@torch.no_grad()
def evaluate(model, tok, val, n_gen=150):
    model.eval()
    losses = [model(input_ids=i, labels=l, attention_mask=a).loss.item() for i, l, a in batches(val, tok, False)]
    correct = 0
    for row in val[:n_gen]:
        x = tok(prompt_of(row["q"]), return_tensors="pt", add_special_tokens=False).to("cuda")
        out = model.generate(**x, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        correct += text == row["a"]
    model.train()
    return sum(losses) / len(losses), correct / min(n_gen, len(val))


def main():
    train, val = load()
    print("train %d, validation %d" % (len(train), len(val)), flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    loss0, em0 = evaluate(model, tok, val, n_gen=60)
    print("before training: val loss %.3f, val exact %.1f%%" % (loss0, 100 * em0), flush=True)

    steps = EPOCHS * math.ceil(len(train) / BATCH) // ACCUM
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(EPOCHS):
        for k, (ids, lab, att) in enumerate(batches(train, tok, True)):
            loss = model(input_ids=ids, labels=lab, attention_mask=att).loss / ACCUM
            loss.backward()
            if (k + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 25 == 0:
                    print("epoch %d step %d/%d loss %.4f (%.0f s)" % (epoch + 1, step, steps, loss.item() * ACCUM,
                                                                      time.time() - t0), flush=True)
        vl, em = evaluate(model, tok, val)
        print("after epoch %d: val loss %.3f, val exact %.1f%%" % (epoch + 1, vl, 100 * em), flush=True)

    merged = model.merge_and_unload()
    out = os.path.join(HERE, "out", "merged")
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("saved merged model to", out, "training time %.0f s, peak GPU memory %.2f GB"
          % (time.time() - t0, torch.cuda.max_memory_allocated() / 1e9), flush=True)


if __name__ == "__main__":
    main()

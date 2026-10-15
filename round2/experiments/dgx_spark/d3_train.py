"""Fine-tune a Decision 3.0 (d3) model on the realistic email training split (runs on the DGX Spark).

d3 ships inference code only. This trainer reuses the model's own prompt builder (D3.prepare) and
replicates its scoring path with gradients: input embeddings -> backbone layers with the model's
non-causal attention (the runtime's PromptEncoder, plus per-layer gradient checkpointing) -> last
position -> FP32 255-way answer-code readout, softmax over the question's valid options. Choice
questions are also trained with their options reversed, because the runtime averages both orders.

Modes, matching the closest recipe already used in this benchmark:
  full  like Laya: all backbone weights and the readout, AdamW (2.5e-5 backbone, 1e-4 readout,
        weight decay 0.01), cosine to 1e-6, 32 items per update, 4 epochs, best validation epoch kept.
  lora  like the Qwen QLoRA: LoRA r=16/alpha=32/dropout 0.05 on every linear layer of the language
        model, readout frozen, AdamW 2e-4 with 10 warm-up updates then cosine to 10%, 8 emails
        (24 forward-order items plus reversed ones) per update, 1 epoch. BF16 base, not 4-bit.
A self-check compares this forward pass with the runtime's own probabilities before training.
The test split is never read. Output: a loadable package (original files, new weights and readout,
no MODEL_MANIFEST.json) plus train_report.json.

Usage: python d3_train.py --repo vllm-sr/d3-edge --revision <sha> --mode full --out DIR --data DIR
"""
import argparse, hashlib, importlib.metadata, importlib.util, json, math, os, random, shutil, sys, time, traceback
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--revision", required=True)
ap.add_argument("--mode", choices=["full", "lora"], required=True)
ap.add_argument("--out", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--micro", type=int, default=4); ap.add_argument("--budget", type=float, default=6 * 3600)
a = ap.parse_args()
OUT, DATA = Path(a.out), Path(a.data)
OUT.mkdir(parents=True, exist_ok=True)
FIELDS = ["category", "priority", "needs_reply"]
torch.manual_seed(42); random.seed(42)
report = {"status": "running", "stage": "load", "repo": a.repo, "revision": a.revision, "mode": a.mode,
          "started_at": time.time(), "gpu": torch.cuda.get_device_name(0),
          "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft"] if importlib.util.find_spec(p.replace("-", "_"))},
          "epochs": [], "loss_by_update": []}


def save():
    (OUT / "train_report.json").write_text(json.dumps(report, indent=1, default=str))


save()
try:
    from transformers import AutoModel
    wrapper = AutoModel.from_pretrained(a.repo, revision=a.revision, trust_remote_code=True, device="cuda:0")
    d3 = wrapper._require()
    backbone, readout0 = d3.backbone, d3.readout
    assert d3.encoder is not None, "expected the non-causal PromptEncoder"
    report["permutation_average"] = bool(d3.permutation_average)
    report["temperature"] = d3.temperature
    root = Path(d3.root)
    questions = json.loads((DATA / "questions.json").read_text())

    def items_for(row):
        prep = d3.prepare(row["state"], questions)
        out = []
        for k in FIELDS:
            q = prep.questions[k]
            keys = list(q.keys)
            want = str(row["expected"][k]).lower() if k == "needs_reply" else row["expected"][k]
            idx = keys.index(want)
            out.append((prep.sequences[k], len(keys), idx, k, False))
            if k in prep.reversed_sequences:
                out.append((prep.reversed_sequences[k], len(keys), len(keys) - 1 - idx, k, True))
        return out

    def encode(ids, mask, train):
        enc = d3.encoder
        model = enc.language_model
        rows, width = mask.shape
        embeds = backbone.get_input_embeddings()(ids)
        ramp = torch.arange(width, device=ids.device).view(1, 1, -1).expand(4, rows, -1)
        text_pos, mrope = ramp[0], ramp[1:]
        rope = model.rotary_emb(embeds, mrope)
        keys = mask.bool()[:, None, None, :]
        scan = enc.recurrent_mask(config=model.config, inputs_embeds=embeds, attention_mask=mask,
                                  past_key_values=None, position_ids=text_pos)
        h = embeds
        for layer, softmax in enc.stack:
            kw = dict(position_embeddings=rope, attention_mask=keys if softmax else scan,
                      position_ids=text_pos, past_key_values=None, use_cache=False)
            h = checkpoint(layer, h, use_reentrant=False, **kw) if train else layer(h, **kw)
        return model.norm(h)[:, -1]

    def code_logits(batch, readout, train):
        width = max(len(s) for s, *_ in batch)
        ids = torch.full((len(batch), width), d3.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, (s, *_r) in enumerate(batch):
            ids[i, width - len(s):] = torch.as_tensor(s); mask[i, width - len(s):] = 1
        ids, mask = ids.cuda(), mask.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = encode(ids, mask, train)
        logits = hidden.float() @ readout.T
        counts = torch.tensor([c for _, c, *_r in batch], device="cuda")[:, None]
        return logits.masked_fill(torch.arange(logits.size(1), device="cuda")[None] >= counts, float("-inf")) / d3.temperature

    # Self-check: this forward pass must reproduce the runtime's probabilities.
    report["stage"] = "self_check"; save()
    rows = [json.loads(l) for l in open(DATA / "train.jsonl")]
    probe = rows[0]
    official = d3.system_one(state=probe["state"], questions=questions)["answers"]
    backbone.eval()
    with torch.no_grad():
        its = items_for(probe)
        p = torch.softmax(code_logits(its, readout0, False), -1)
    mine = {}
    for (s, c, idx, k, rev), row in zip(its, p):
        mine.setdefault(k, []).append((rev, row[:c].tolist()))
    diffs = []
    for k in FIELDS:
        fwd = [r for rev, r in mine[k] if not rev][0]
        bwd = [r for rev, r in mine[k] if rev]
        probs = [(x + y) / 2 for x, y in zip(fwd, reversed(bwd[0]))] if bwd and d3.permutation_average else fwd
        ans = official[k]
        if "noul" in ans:
            diffs.append(abs(probs[-1] - ans["noul"]))
        else:
            diffs.append(max(abs(x - y) for x, y in zip(probs, ans["probabilities"].values())))
    report["self_check_max_abs_diff"] = max(diffs)
    save()
    assert max(diffs) < 0.02, f"forward pass does not match the runtime: {diffs}"

    report["stage"] = "prepare"; save()
    train_items = [it for r in rows for it in items_for(r)]
    val_rows = [json.loads(l) for l in open(DATA / "validation.jsonl")]
    report.update(train_rows=len(rows), train_items=len(train_items), validation_rows=len(val_rows))

    if a.mode == "full":
        backbone.float()
        for p_ in backbone.parameters():
            p_.requires_grad = True
        readout = torch.nn.Parameter(readout0.detach().clone().float())
        groups = [{"params": [p_ for p_ in backbone.parameters()], "lr": 2.5e-5, "weight_decay": 0.01},
                  {"params": [readout], "lr": 1e-4, "weight_decay": 0.01}]
        epochs, per_update = 4, 32
        base_lrs = [2.5e-5, 1e-4]
    else:
        from peft import LoraConfig, get_peft_model
        for p_ in backbone.parameters():
            p_.requires_grad = False
        lm = d3.encoder.language_model
        leaf = sorted({n.split(".")[-1] for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear)})
        pattern = r"^(?!.*(visual|mtp)).*\.(" + "|".join(leaf) + r")$"
        peft_model = get_peft_model(backbone, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=pattern))
        readout = readout0.detach().float()
        groups = [{"params": [p_ for p_ in peft_model.parameters() if p_.requires_grad], "lr": 2e-4, "weight_decay": 0.0}]
        epochs, per_update = 1, 24
        base_lrs = [2e-4]
        report["lora_targets"] = leaf
    trainable = sum(p_.numel() for g in groups for p_ in g["params"])
    report["trainable_parameters"] = trainable
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    acc_steps = max(1, per_update // a.micro)
    steps_per_epoch = math.ceil(len(train_items) / a.micro)
    total_updates = math.ceil(steps_per_epoch / acc_steps) * epochs
    report.update(epochs_planned=epochs, items_per_update=a.micro * acc_steps, total_updates_planned=total_updates)

    def lr_scale(u):
        if a.mode == "full":
            frac = min(1.0, u / max(1, total_updates))
            return None, frac
        if u < 10:
            return (u + 1) / 10, None
        frac = min(1.0, (u - 10) / max(1, total_updates - 10))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)), None

    def set_lr(u):
        s, frac = lr_scale(u)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = (1e-6 + (base - 1e-6) * 0.5 * (1 + math.cos(math.pi * frac))) if a.mode == "full" else base * s

    def validate(readout_t):
        backbone.eval(); correct = 0; per = {k: 0 for k in FIELDS}
        with torch.no_grad():
            for r in val_rows:
                its = items_for(r)
                pr = torch.softmax(code_logits(its, readout_t, False), -1)
                got = {}
                for (s, c, idx, k, rev), row in zip(its, pr):
                    v = row[:c]
                    if rev:
                        v = v.flip(0)
                    got.setdefault(k, []).append((v, idx if not rev else c - 1 - idx))
                ok = True
                for k in FIELDS:
                    avg = sum(v for v, _ in got[k]) / len(got[k])
                    hit = int(avg.argmax()) == got[k][0][1]
                    per[k] += hit; ok &= hit
                correct += ok
        backbone.train()
        return {"all_fields_accuracy": correct / len(val_rows), "per_field_accuracy": {k: v / len(val_rows) for k, v in per.items()}}

    report["baseline_validation"] = validate(readout); save()
    report["stage"] = "training"; save()
    backbone.train(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter(); u = 0; best = -1.0; stopped_early = False

    def save_package(tag_dir):
        tag_dir.mkdir(parents=True, exist_ok=True)
        for f in root.iterdir():
            if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("MODEL_MANIFEST.json", "model.safetensors.index.json"):
                shutil.copy2(f, tag_dir / f.name)
        for sub in ["assets"]:
            if (root / sub).is_dir() and not (tag_dir / sub).exists():
                shutil.copytree(root / sub, tag_dir / sub)
        from safetensors import safe_open
        from safetensors.torch import save_file
        index = json.loads((root / "model.safetensors.index.json").read_text()) if (root / "model.safetensors.index.json").exists() else None
        files = sorted(set(index["weight_map"].values())) if index else ["model.safetensors"]
        src = backbone if a.mode == "full" else peft_model.merge_and_unload(progressbar=False, safe_merge=True)
        state = {k: v for k, v in src.state_dict().items()}
        written, missing = set(), []
        for fname in files:
            with safe_open(str(root / fname), framework="pt") as h:
                names = list(h.keys()); dtypes = {n: h.get_slice(n).get_dtype() for n in names}
            shard = {}
            for n in names:
                cand = [n, n.replace("model.", "", 1), "model." + n]
                key = next((c for c in cand if c in state), None)
                if key is None:
                    missing.append(n); continue
                shard[n] = state[key].detach().to(torch.bfloat16 if dtypes[n] == "BF16" else state[key].dtype).contiguous().cpu()
                written.add(n)
            save_file(shard, str(tag_dir / fname), metadata={"format": "pt"})
        assert not missing, f"weights not found in the trained model: {missing[:5]}"
        if index:
            shutil.copy2(root / "model.safetensors.index.json", tag_dir / "model.safetensors.index.json")
        save_file({"weight": readout.detach().float().contiguous().cpu()}, str(tag_dir / "readout.safetensors"))
        man = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in tag_dir.iterdir() if f.is_file() and f.suffix == ".safetensors"}
        (tag_dir / "FINETUNE_MANIFEST.json").write_text(json.dumps({"base": a.repo, "base_revision": a.revision, "mode": a.mode, "sha256": man}, indent=1))
        return len(written)

    for ep in range(epochs):
        order = list(range(len(train_items))); random.Random(42 + ep).shuffle(order)
        ep_start, run_loss, n_steps = time.perf_counter(), 0.0, 0
        for s0 in range(0, len(order), a.micro):
            batch = [train_items[i] for i in order[s0:s0 + a.micro]]
            logits = code_logits(batch, readout, True)
            target = torch.tensor([idx for _, _, idx, *_r in batch], device="cuda")
            loss = F.cross_entropy(logits, target)
            (loss / acc_steps).backward()
            run_loss += loss.item(); n_steps += 1
            if n_steps % acc_steps == 0:
                set_lr(u)
                torch.nn.utils.clip_grad_norm_([p_ for g in groups for p_ in g["params"]], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); u += 1
                if u % 20 == 0:
                    report["loss_by_update"].append({"update": u, "mean_loss": run_loss / n_steps, "elapsed": time.perf_counter() - start})
                    save(); print("D3_TRAIN", json.dumps(report["loss_by_update"][-1]), flush=True)
            if time.perf_counter() - start > a.budget:
                stopped_early = True; break
        ep_row = {"epoch": ep + 1, "mean_loss": run_loss / max(1, n_steps), "training_seconds": time.perf_counter() - ep_start}
        v0 = time.perf_counter(); ep_row["validation"] = validate(readout); ep_row["validation_seconds"] = time.perf_counter() - v0
        report["epochs"].append(ep_row); save(); print("D3_EPOCH", json.dumps(ep_row), flush=True)
        if a.mode == "full" and ep_row["validation"]["all_fields_accuracy"] > best:
            best = ep_row["validation"]["all_fields_accuracy"]
            report["selected_epoch"] = ep + 1
            report["weights_written"] = save_package(OUT / "package")
            save()
        if stopped_early:
            break
    report["training_seconds"] = time.perf_counter() - start
    report["peak_training_gpu_bytes"] = torch.cuda.max_memory_allocated()
    report["stopped_early_by_budget"] = stopped_early
    if a.mode == "lora":
        report["selected_epoch"] = len(report["epochs"])
        report["weights_written"] = save_package(OUT / "package")
    report["status"] = "complete"; report["stage"] = "complete"; report["finished_at"] = time.time(); save()
    print("D3_TRAIN_COMPLETE", json.dumps({k: report.get(k) for k in ["mode", "selected_epoch", "training_seconds", "weights_written"]}), flush=True)
except BaseException:
    report["status"] = "failed"; report["error"] = traceback.format_exc(); save(); print(report["error"], flush=True); sys.exit(1)

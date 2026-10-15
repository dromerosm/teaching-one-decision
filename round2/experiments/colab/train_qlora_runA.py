"""QLoRA fine-tune of Qwen3.5-9B on the same training split Laya used.

Recipe: 4-bit NF4 base (same quantization as the Qwen evaluation), LoRA r=16 / alpha=32 /
dropout 0.05 on every linear layer of the language model (vision tower and MTP head
excluded), loss only on the answer tokens, paged 8-bit AdamW, lr 2e-4 with 10-update
warm-up then cosine to 10%, effective batch 8, at most one epoch. The prompt is prompt v2
(qwen_common.py), identical to the qwen_prompt evaluation, so the only difference between
qwen_prompt and qwen_qlora is the adapter. Training is time-capped by QLORA_BUDGET_SECONDS;
if one epoch does not fit, the run trains on as many shuffled examples as fit and says so.
Validation, calibration and test files are never read.
"""
import os
os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
import hashlib, importlib.metadata, json, math, random, sys, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path("/content/laya-finetune"); DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
from qwen_common import system_prompt, target_json

REPO = "Qwen/Qwen3.5-9B"; REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BUDGET = float(os.environ.get("QLORA_BUDGET_SECONDS", "9900"))
ACC, LR, WARMUP = 8, 2e-4, 10
report = {"status": "running", "stage": "preflight", "started_at": time.time(), "repo": REPO, "revision": REV,
          "budget_seconds": BUDGET, "recipe": {"quantization": "bnb NF4 double-quant, fp16 compute", "lora_r": 16,
          "lora_alpha": 32, "lora_dropout": 0.05, "targets": "all linear layers of the language model",
          "loss": "cross-entropy on answer tokens only", "optimizer": "bnb PagedAdamW8bit", "lr": LR,
          "schedule": f"{WARMUP}-update linear warm-up, cosine to 10%", "effective_batch": ACC, "max_epochs": 1,
          "seed": 42, "prompt": "qwen_common.SYSTEM_V2 + schema"}, "loss_by_update": []}


def save():
    tmp = ROOT / "qlora_report.tmp"; tmp.write_text(json.dumps(report, indent=2, default=str)); tmp.replace(ROOT / "qlora_report.json")


save()
try:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    import bitsandbytes as bnb
    report["versions"] = {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "bitsandbytes", "flash-linear-attention"]}
    assert "T4" in torch.cuda.get_device_name(0); report["gpu"] = torch.cuda.get_device_name(0)
    torch.manual_seed(42); random.seed(42)
    report["stage"] = "download"; save(); t = time.perf_counter()
    path = snapshot_download(REPO, revision=REV, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
    report["download_seconds"] = time.perf_counter() - t
    report["stage"] = "load"; save(); t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(
        path, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        device_map={"": "cuda:0"}, dtype=torch.float16, attn_implementation="sdpa")
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    leaf = sorted({n.split(".")[-1] for n, m in model.named_modules()
                   if isinstance(m, bnb.nn.Linear4bit) and "visual" not in n and "mtp" not in n and "lm_head" not in n})
    assert leaf, "no 4-bit linear layers found"
    pattern = r"^(?!.*(visual|mtp)).*\.(" + "|".join(leaf) + r")$"
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=pattern))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    report.update(target_leaf_modules=leaf, lora_modules=sum(1 for n, _ in model.named_modules() if n.endswith("lora_A")),
                  trainable_parameters=trainable, total_parameters_reported=total, load_seconds=time.perf_counter() - t)
    report["stage"] = "tokenize"; save()
    schema = json.loads((DATA / "schema.json").read_text())
    system = system_prompt(schema)
    rows = [json.loads(l) for l in open(DATA / "train.jsonl")]
    items = []
    for r in rows:
        p = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": r["state"]}],
                                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True)["input_ids"]
        if p and isinstance(p[0], list):
            p = p[0]
        tgt = tok.encode(target_json(r["expected"]), add_special_tokens=False) + [tok.eos_token_id]
        items.append((list(p), tgt))
    lens = [len(p) + len(t) for p, t in items]
    report.update(train_rows=len(items), tokens_per_example={"min": min(lens), "max": max(lens), "mean": sum(lens) / len(lens)},
                  train_sha256=hashlib.sha256((DATA / "train.jsonl").read_bytes()).hexdigest())
    order = list(range(len(items))); random.Random(42).shuffle(order)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    planned = len(items); total_updates = math.ceil(planned / ACC)

    def lr_at(u):
        if u < WARMUP:
            return LR * (u + 1) / WARMUP
        frac = min(1.0, (u - WARMUP) / max(1, total_updates - WARMUP))
        return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    report["stage"] = "training"; save()
    model.train(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter(); running = 0.0; u = 0; seen = 0; skipped_steps = 0
    for k, idx in enumerate(order):
        if k == 16:  # fix the plan from measured speed (includes two optimizer steps)
            per = (time.perf_counter() - start) / 16
            planned = min(len(items), int(BUDGET / per) // ACC * ACC)
            total_updates = planned // ACC
            report.update(seconds_per_example_first16=per, planned_examples=planned, planned_updates=total_updates,
                          full_epoch=planned == len(items)); save()
        if k >= planned or time.perf_counter() - start > BUDGET * 1.05:
            break
        p, tgt = items[idx]
        ids = torch.tensor([p + tgt], device="cuda"); K = len(tgt)
        with torch.autocast("cuda", dtype=torch.float16):
            try:
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=K + 1)
            except TypeError:
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
        logits = out.logits
        pred = logits[:, :K] if logits.size(1) == K + 1 else logits[:, -K - 1:-1]
        loss = F.cross_entropy(pred.float().reshape(-1, pred.size(-1)), torch.tensor(tgt, device="cuda"))
        scaler.scale(loss / ACC).backward()
        running += loss.item(); seen += 1
        if seen % ACC == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(u)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            before = scaler.get_scale(); scaler.step(opt); scaler.update()
            skipped_steps += scaler.get_scale() < before
            opt.zero_grad(set_to_none=True)
            report["loss_by_update"].append({"update": u + 1, "mean_loss": running / ACC, "lr": lr_at(u),
                                             "elapsed": time.perf_counter() - start})
            running = 0.0; u += 1
            if u % 10 == 0:
                save(); print("QLORA_UPDATE " + json.dumps(report["loss_by_update"][-1]), flush=True)
    torch.cuda.synchronize()
    report.update(training_seconds=time.perf_counter() - start, examples_seen=seen, updates=u,
                  scaler_skipped_steps_upper_bound=skipped_steps, epochs_fraction=seen / len(items),
                  peak_training_gpu_bytes=torch.cuda.max_memory_allocated())
    report["stage"] = "save"; save()
    out_dir = ROOT / "qlora_adapter"
    model.save_pretrained(str(out_dir))
    report["adapter_files"] = {f.name: {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                               for f in out_dir.iterdir() if f.is_file()}
    report["status"] = "complete"; report["stage"] = "complete"; report["finished_at"] = time.time(); save()
    print("QLORA_COMPLETE " + json.dumps({k: report[k] for k in ["examples_seen", "updates", "training_seconds", "epochs_fraction"]}), flush=True)
except BaseException:
    report["status"] = "failed"; report["error"] = traceback.format_exc(); report["finished_at"] = time.time(); save()
    print(report["error"], flush=True); raise

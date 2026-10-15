"""Diagnose why d3 training-mode loss starts at chance: compare one batch's loss across modes."""
import json, sys, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
DATA = Path(sys.argv[1]); FIELDS = ["category", "priority", "needs_reply"]
w = AutoModel.from_pretrained("vllm-sr/d3-edge", revision="20c745653400d191ecacd99c4eafc0c5900dc15e", trust_remote_code=True, device="cuda:0")
d3 = w._require(); bb = d3.backbone; q = json.loads((DATA / "questions.json").read_text())
rows = [json.loads(l) for l in open(DATA / "train.jsonl")][:16]
items = []
for r in rows:
    p = d3.prepare(r["state"], q)
    for k in FIELDS:
        keys = list(p.questions[k].keys); want = str(r["expected"][k]).lower() if k == "needs_reply" else r["expected"][k]
        items.append((p.sequences[k], len(keys), keys.index(want)))
print("dtype", next(bb.parameters()).dtype, "training flag", bb.training, "layer types", type(d3.encoder.stack[0][0]).__name__)
def enc(ids, mask, ckpt):
    e = d3.encoder; m = e.language_model; rows_, width = mask.shape
    emb = bb.get_input_embeddings()(ids)
    ramp = torch.arange(width, device=ids.device).view(1, 1, -1).expand(4, rows_, -1)
    rope = m.rotary_emb(emb, ramp[1:]); keys = mask.bool()[:, None, None, :]
    scan = e.recurrent_mask(config=m.config, inputs_embeds=emb, attention_mask=mask, past_key_values=None, position_ids=ramp[0])
    h = emb
    for layer, soft in e.stack:
        kw = dict(position_embeddings=rope, attention_mask=keys if soft else scan, position_ids=ramp[0], past_key_values=None, use_cache=False)
        h = checkpoint(layer, h, use_reentrant=False, **kw) if ckpt else layer(h, **kw)
    return m.norm(h)[:, -1]
def loss(batch, ckpt, autocast, grad):
    width = max(len(s) for s, *_ in batch)
    ids = torch.full((len(batch), width), d3.pad_id); mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (s, *_r) in enumerate(batch):
        ids[i, width - len(s):] = torch.as_tensor(s); mask[i, width - len(s):] = 1
    ids, mask = ids.cuda(), mask.cuda()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.autocast("cuda", enabled=False)
    with torch.set_grad_enabled(grad), ctx:
        h = enc(ids, mask, ckpt)
    lg = h.float() @ d3.readout.float().T
    cnt = torch.tensor([c for _, c, _ in batch], device="cuda")[:, None]
    lg = lg.masked_fill(torch.arange(lg.size(1), device="cuda")[None] >= cnt, float("-inf"))
    t = torch.tensor([x for _, _, x in batch], device="cuda")
    return round(F.cross_entropy(lg, t).item(), 4), round((lg.argmax(-1) == t).float().mean().item(), 3)
for bs in [1, 4]:
    batch = items[:bs] if bs == 1 else items[:12]
    for train_mode in [False, True]:
        bb.train(train_mode)
        for ckpt in [False, True]:
            for grad in [False, True]:
                if ckpt and not grad: continue
                print(f"bs={len(batch)} train={train_mode} ckpt={ckpt} grad={grad} bf16-native:", loss(batch, ckpt, False, grad), flush=True)
bb.float(); bb.train(True)
print("after float(): train ckpt grad autocast:", loss(items[:12], True, True, True), " eval nograd autocast:", (bb.eval(), loss(items[:12], False, True, False))[1])

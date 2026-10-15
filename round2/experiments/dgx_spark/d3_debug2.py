"""Overfit test: can d3-edge drive the loss of one fixed batch to ~0? Compare fla kernels vs torch reference."""
import json, sys, copy, torch, torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel
import transformers.models.qwen3_5.modeling_qwen3_5 as qm
DATA = Path(sys.argv[1]); FIELDS = ["category", "priority", "needs_reply"]
q = json.loads((DATA / "questions.json").read_text())
rows = [json.loads(l) for l in open(DATA / "train.jsonl")][:4]
def run(impl, ckpt):
    w = AutoModel.from_pretrained("vllm-sr/d3-edge", revision="20c745653400d191ecacd99c4eafc0c5900dc15e", trust_remote_code=True, device="cuda:0")
    d3 = w._require(); bb = d3.backbone
    if impl == "torch":
        n = 0
        for m in bb.modules():
            if hasattr(m, "chunk_gated_delta_rule"):
                m.chunk_gated_delta_rule = qm.torch_chunk_gated_delta_rule; n += 1
        print("patched", n, "GatedDeltaNet modules to torch reference")
    items = []
    for r in rows:
        p = d3.prepare(r["state"], q)
        for k in FIELDS:
            keys = list(p.questions[k].keys); want = str(r["expected"][k]).lower() if k == "needs_reply" else r["expected"][k]
            items.append((p.sequences[k], len(keys), keys.index(want)))
    bb.float(); bb.train()
    for p_ in bb.parameters(): p_.requires_grad = True
    ro = torch.nn.Parameter(d3.readout.detach().float().clone())
    opt = torch.optim.AdamW([{"params": list(bb.parameters()), "lr": 2.5e-5}, {"params": [ro], "lr": 1e-4}])
    from torch.utils.checkpoint import checkpoint
    e = d3.encoder; m = e.language_model
    width = max(len(s) for s, *_ in items)
    ids = torch.full((len(items), width), d3.pad_id); mask = torch.zeros((len(items), width), dtype=torch.long)
    for i, (s, *_r) in enumerate(items):
        ids[i, width - len(s):] = torch.as_tensor(s); mask[i, width - len(s):] = 1
    ids, mask = ids.cuda(), mask.cuda()
    t = torch.tensor([x for *_a, x in items], device="cuda"); cnt = torch.tensor([c for _, c, _ in items], device="cuda")[:, None]
    out = []
    for step in range(40):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = bb.get_input_embeddings()(ids)
            ramp = torch.arange(width, device="cuda").view(1, 1, -1).expand(4, len(items), -1)
            rope = m.rotary_emb(emb, ramp[1:]); keys = mask.bool()[:, None, None, :]
            scan = e.recurrent_mask(config=m.config, inputs_embeds=emb, attention_mask=mask, past_key_values=None, position_ids=ramp[0])
            h = emb
            for layer, soft in e.stack:
                kw = dict(position_embeddings=rope, attention_mask=keys if soft else scan, position_ids=ramp[0], past_key_values=None, use_cache=False)
                h = checkpoint(layer, h, use_reentrant=False, **kw) if ckpt else layer(h, **kw)
            hid = m.norm(h)[:, -1]
        lg = (hid.float() @ ro.T).masked_fill(torch.arange(255, device="cuda")[None] >= cnt, float("-inf"))
        loss = F.cross_entropy(lg, t); opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(bb.parameters()) + [ro], 1.0)
        opt.step()
        if step % 5 == 0 or step == 39: out.append((step, round(loss.item(), 4), round(float(gn), 2)))
    print(impl, "ckpt" if ckpt else "nockpt", out, flush=True)
    del w, d3, bb, opt; torch.cuda.empty_cache()
run("fla", True)
run("fla", False)
run("torch", False)

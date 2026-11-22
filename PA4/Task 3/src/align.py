"""Step 5: distribution alignment of every draft with the target on common, frozen prefixes.

For each bank, prompt and saved position pos, the prefix h = chat-template(prompt) + response[:pos-1]. One target
forward and one forward per draft over chat-template(prompt) + response[:max_pos-1] give p_T(.|h) and q(.|h) at all
saved positions of that prompt (FP32 log-softmax over the full 151,936-token vocabulary, temperature 1). Per position:
    forward KL  sum p log(p/q)      reverse KL  sum q log(q/p)      TV  0.5 sum |p - q|
    expected one-token acceptance 1 - TV        greedy agreement  argmax p == argmax q
plus the target entropy (difficulty of the position), the target / draft argmax tokens and probabilities (for aligned
prefix-level disagreements). Every draft is evaluated on the same prefix IDs and positions, so all averages share
token-weighted denominators.
"""
from __future__ import annotations

import time

import torch

from .data import encode
from .utils import log, save_jsonl_gz


def pos_bin(pos: int) -> str:
    return "1-16" if pos <= 16 else "17-32" if pos <= 32 else "33-64" if pos <= 64 else "65-192"


@torch.no_grad()
def evaluate_banks(banks: dict, target, drafts: dict, tok, path, dev="cuda") -> dict:
    """banks: name -> bank dict (records); drafts: name -> model. Writes one row per (bank, prompt, position)."""
    rows = []
    t0 = time.perf_counter()
    for bname, bank in banks.items():
        for rec in bank["records"]:
            if not rec["positions"]:
                continue
            prompt = encode(tok, rec["prompt"])
            maxp = max(rec["positions"])
            ids = torch.tensor([prompt + rec["response_ids"][:maxp - 1]], device=dev)
            idx = torch.tensor([len(prompt) - 1 + pos - 1 for pos in rec["positions"]], device=dev)
            lp = torch.log_softmax(target(input_ids=ids).logits[0, idx].float(), -1)
            p = lp.exp()
            ent = -(p * lp).sum(-1)
            pmax, parg = p.max(-1)
            per = {}
            for dname, dm in drafts.items():
                lq = torch.log_softmax(dm(input_ids=ids).logits[0, idx].float(), -1)
                q = lq.exp()
                qmax, qarg = q.max(-1)
                per[dname] = {"fkl": (p * (lp - lq)).sum(-1), "rkl": (q * (lq - lp)).sum(-1),
                              "tv": 0.5 * (p - q).abs().sum(-1), "agree": (parg == qarg).float(),
                              "q_argmax": qarg, "q_argmax_prob": qmax, "p_of_q_argmax": p.gather(1, qarg[:, None])[:, 0]}
            for j, pos in enumerate(rec["positions"]):
                row = {"bank": bname, "producer": bank["producer"], "prompt_id": rec["prompt_id"],
                       "workload": rec["workload"], "position": pos, "bin": pos_bin(pos),
                       "target_entropy": float(ent[j]), "p_argmax": int(parg[j]), "p_argmax_prob": float(pmax[j]),
                       "next_token_in_bank": int(rec["response_ids"][pos - 1]) if pos - 1 < len(rec["response_ids"]) else None}
                for dname, m in per.items():
                    row[dname] = {k: (float(v[j]) if k not in ("q_argmax",) else int(v[j])) for k, v in m.items()}
                    row[dname]["exp_acc"] = 1.0 - row[dname]["tv"]
                rows.append(row)
        log(f"align: bank {bname} done ({len(rows)} rows so far)")
    save_jsonl_gz(path, rows)
    return {"rows": len(rows), "seconds": time.perf_counter() - t0, "banks": list(banks), "drafts": list(drafts)}

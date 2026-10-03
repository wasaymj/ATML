from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def get_batch_logps(model, batch):
    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()
    response_mask = batch["response_mask"].cuda()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    labels = input_ids[:, 1:]
    logits = logits[:, :-1, :]
    # Cast to float32 before log_softmax: computing softmax over a ~150k vocabulary
    # in float16 causes severe underflow/overflow, leading to NaN loss values.
    # The course's own generation.py (response_sequence_logprobs) applies the same cast.
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    token_logps = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = response_mask[:, 1:].float()
    return (token_logps * mask).sum(dim=-1)


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    import json
    from common.models import reference_mode

    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model = bundle["model"]
    optimizer = bundle["optimizer"]
    loader = bundle["loader"]
    beta_val = bundle["beta"]
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    epochs = int(cfg.get("epochs", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    optimizer.zero_grad()
    step = 0
    logs = []

    for epoch in range(epochs):
        for i, (chosen_batch, rejected_batch) in enumerate(loader):
            with torch.no_grad():
                with reference_mode(model):
                    ref_chosen_logp = get_batch_logps(model, chosen_batch)
                    ref_rejected_logp = get_batch_logps(model, rejected_batch)

            policy_chosen_logp = get_batch_logps(model, chosen_batch)
            policy_rejected_logp = get_batch_logps(model, rejected_batch)

            loss, metrics = dpo_loss(
                policy_chosen_logp, policy_rejected_logp,
                ref_chosen_logp, ref_rejected_logp, beta_val
            )

            (loss / grad_accum_steps).backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                logit_vals = beta_val * (
                    (policy_chosen_logp - policy_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
                ).detach()
                print(f"Epoch {epoch} Step {step}/{len(loader)//grad_accum_steps} Loss {loss.item():.4f} Acc {metrics['preference_accuracy'].item():.4f} LogitMean {logit_vals.mean().item():.4f}")
                logs.append({
                    "step": step,
                    "loss": loss.item(),
                    "preference_accuracy": metrics["preference_accuracy"].item(),
                    "policy_margin": metrics["policy_margin_mean"].item(),
                    "logit_mean": metrics["logit_mean"].item(),
                    "logit_std": logit_vals.std().item(),
                })

    model.save_pretrained(output)
    with open(output / "logs.json", "w") as f:
        json.dump(logs, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()

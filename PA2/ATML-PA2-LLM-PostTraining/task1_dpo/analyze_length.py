from __future__ import annotations

import argparse
from common.data import load_yaml, read_jsonl, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))
    import subprocess
    import torch
    import numpy as np
    from common.models import reference_mode, load_policy, load_tokenizer
    from common.data import prompt_messages_from_preference, encode_prompt_response, pad_batch, preference_responses, prompt_messages
    from common.generation import response_sequence_logprobs, batch_generate
    from common.metrics import word_limit_compliance
    from tqdm import tqdm

    balanced_run_name = "length_balanced"
    balanced_output_dir = "outputs/task1_dpo/length_balanced"
    
    print("\n--- Training Length-Balanced DPO ---")
    train_cmd = [
        "python", "-m", "task1_dpo.train",
        "--config", args.config,
        "--run-name", balanced_run_name,
        "--output", balanced_output_dir,
        "--dataset", cfg["paths"]["dpo_length_train"]
    ]
    print("Running:", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True)

    print("\n--- Evaluating Models on Length Strata and Word Limits ---")
    models_to_eval = {
        "Standard DPO": "outputs/task1_dpo/standard",
        "Length-Balanced DPO": balanced_output_dir
    }
    
    tokenizer = load_tokenizer(cfg["base_model"])
    max_seq_length = int(cfg["max_sequence_length"])
    batch_size = int(cfg["batch_size"])
    
    def make_collate(tokenizer, max_length):
        def collate(rows):
            chosen, rejected = [], []
            for row in rows:
                prompt = prompt_messages_from_preference(row)
                yc, yr = preference_responses(row)
                chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
                rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
            return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected), rows
        return collate
        
    loader = torch.utils.data.DataLoader(stratified, batch_size=batch_size, shuffle=False, collate_fn=make_collate(tokenizer, max_seq_length))
    word_limit_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    
    for model_name, adapter_path in models_to_eval.items():
        print(f"\nEvaluating {model_name} from {adapter_path}...")
        policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
        policy.eval()
        
        strata_correct = {"preferred_longer": [], "length_matched": [], "rejected_longer": []}
        
        for chosen_batch, rejected_batch, batch_rows in tqdm(loader, desc=f"{model_name} Preference Acc"):
            chosen_batch = {k: v.cuda() for k, v in chosen_batch.items() if isinstance(v, torch.Tensor)}
            rejected_batch = {k: v.cuda() for k, v in rejected_batch.items() if isinstance(v, torch.Tensor)}
            
            with torch.no_grad():
                policy_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
                policy_rejected_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)

                with reference_mode(policy):
                    ref_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
                    ref_rejected_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)

                policy_margin = policy_chosen_logp - policy_rejected_logp
                ref_margin = ref_chosen_logp - ref_rejected_logp
                
                corrects = (policy_margin - ref_margin > 0).cpu().numpy()
                
                for idx, row in enumerate(batch_rows):
                    strata_correct[row["length_stratum"]].append(corrects[idx])
        
        prompts = [prompt_messages(r) for r in word_limit_rows]
        compliance_scores = []
        gen_lengths = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name} Generation"):
            batch_prompts = prompts[i:i+batch_size]
            batch_rows = word_limit_rows[i:i+batch_size]
            
            gen_out = batch_generate(
                policy, tokenizer, batch_prompts,
                max_prompt_length=max_seq_length - int(cfg["max_generation_tokens"]),
                max_new_tokens=int(cfg["max_generation_tokens"]),
                temperature=float(cfg.get("temperature", 0.7)),
                top_p=float(cfg.get("top_p", 0.9)),
                do_sample=True
            )
            
            gen_lengths.extend(gen_out["response_lengths"])
            for r, response in zip(batch_rows, gen_out["responses"]):
                # word_limit_prompts has no 'prompt' key; the user text is in messages[0]['content']
                prompt_text = r["messages"][0]["content"]
                c = word_limit_compliance(prompt_text, response)
                if c is not None:
                    compliance_scores.append(c)
        
        results_summary = {
            "stratum_accuracy": {k: float(np.mean(v)) for k, v in strata_correct.items()},
            "mean_generated_length": float(np.mean(gen_lengths)),
            "word_limit_compliance": float(np.mean(compliance_scores))
        }
        
        print(f"\n--- {model_name} Results ---")
        print(f"Stratum 'preferred_longer' Acc: {np.mean(strata_correct['preferred_longer']):.4f}")
        print(f"Stratum 'length_matched' Acc:   {np.mean(strata_correct['length_matched']):.4f}")
        print(f"Stratum 'rejected_longer' Acc:  {np.mean(strata_correct['rejected_longer']):.4f}")
        print(f"Mean Generated Length:          {np.mean(gen_lengths):.2f}")
        print(f"Word Limit Compliance:          {np.mean(compliance_scores):.4f}")
        
        results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
        results_dir.mkdir(parents=True, exist_ok=True)
        import json
        
        safe_name = model_name.replace(" ", "_").lower()
        with open(results_dir / f"length_analysis_{safe_name}.json", "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"Results saved to {results_dir / f'length_analysis_{safe_name}.json'}")
        
        del policy
        import gc; gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from typing import Dict
from torch.utils.data import DataLoader

@torch.no_grad()
def evaluate_source_domains(
    backbone: nn.Module,
    head: nn.Module,
    val_loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluates the model on each source domain's validation set."""
    backbone.eval()
    head.eval()

    results = {}
    for domain, loader in val_loaders.items():
        all_preds, all_labels = [], []
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)
            logits = head(feats)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
        
        preds_t = torch.cat(all_preds).numpy()
        labels_t = torch.cat(all_labels).numpy()

        acc = (preds_t == labels_t).mean()
        macro_f1 = f1_score(labels_t, preds_t, average="macro", zero_division=0)

        results[domain] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
        }
    
    # Calculate mean and worst domain
    mean_acc = sum(r["accuracy"] for r in results.values()) / len(results)
    mean_f1 = sum(r["macro_f1"] for r in results.values()) / len(results)
    
    worst_domain_f1 = min(results.keys(), key=lambda k: results[k]["macro_f1"])
    worst_f1 = results[worst_domain_f1]["macro_f1"]

    results["mean"] = {"accuracy": mean_acc, "macro_f1": mean_f1}
    results["worst"] = {"domain": worst_domain_f1, "macro_f1": worst_f1}

    return results

import torch
import numpy as np
from sklearn.metrics import f1_score
from torchvision import transforms as T

def prediction_consistency(preds_clean: np.ndarray, preds_transformed: np.ndarray) -> float:
    return float((preds_clean == preds_transformed).mean())

def evaluate(
    model_name : str,
    images     : torch.Tensor,
    labels     : torch.Tensor,
    MODELS     : dict,
    device,
    head       = None,
    batch_size : int = 128,
) -> dict:
    wrapper = MODELS[model_name]
    all_preds, all_probs, all_labels = [], [], []
 
    for i in range(0, len(images), batch_size):
        xb = images[i : i + batch_size]
        lb = labels[i : i + batch_size]
 
        feats  = wrapper.get_features(xb, device)
        if head is None:
            raise ValueError("Linear head must be provided for evaluation.")
        logits = head(feats.to(device))
        probs  = logits.softmax(dim=-1)
 
        all_preds.append(logits.argmax(1).cpu())
        all_probs.append(probs.cpu())
        all_labels.append(lb)
 
    preds     = torch.cat(all_preds).detach().numpy()
    probs_arr = torch.cat(all_probs).detach().numpy()
    labels_np = torch.cat(all_labels).detach().numpy()
 
    acc          = (preds == labels_np).mean()
    macro_f1     = f1_score(labels_np, preds, average="macro", zero_division=0)
    mean_max_conf = probs_arr.max(axis=1).mean()
 
    return {
        "accuracy"       : float(acc),
        "macro_f1"       : float(macro_f1),
        "mean_max_conf"  : float(mean_max_conf),
        "predictions"    : preds,
        "probabilities"  : probs_arr,
    }
    
def evaluate_clip_zeroshot(
    images     : torch.Tensor,
    labels     : torch.Tensor,
    clip_model,
    clip_text_feats,
    device,
    batch_size : int = 128,
) -> dict:
    NORM_clip = T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
    all_preds, all_probs, all_labels = [], [], []
 
    for i in range(0, len(images), batch_size):
        xb = images[i : i + batch_size]
        lb = labels[i : i + batch_size]
        
        x        = NORM_clip(xb.to(device))
        img_f    = clip_model.visual(x)
        img_f    = img_f / img_f.norm(dim=-1, keepdim=True)
     
        scale    = clip_model.logit_scale.exp()
        logits   = scale * img_f @ clip_text_feats.T
        probs    = logits.softmax(dim=-1)
        
        all_preds.append(logits.argmax(1).cpu())
        all_probs.append(probs.cpu())
        all_labels.append(lb)
 
    preds     = torch.cat(all_preds).numpy()
    probs_arr = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
 
    acc           = (preds == labels_np).mean()
    macro_f1      = f1_score(labels_np, preds, average="macro", zero_division=0)
    mean_max_conf = probs_arr.max(axis=1).mean()
 
    return {
        "accuracy"      : float(acc),
        "macro_f1"      : float(macro_f1),
        "mean_max_conf" : float(mean_max_conf),
        "predictions"   : preds,
        "probabilities" : probs_arr,
    }

def eval_shape_bias(model_name, images, content_labels, style_labels, MODELS, device, head=None, clip_model=None, clip_text_feats=None):
    all_preds = []
    wrapper   = MODELS.get(model_name)
    NORM_clip = T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    for i in range(0, len(images), 64):
        xb = images[i:i+64]
        if head is not None:
            feats  = wrapper.get_features(xb, device)
            logits = head(feats.to(device))
        else:
            x        = NORM_clip(xb.to(device))
            img_f    = clip_model.visual(x)
            img_f    = img_f / img_f.norm(dim=-1, keepdim=True)
            scale    = clip_model.logit_scale.exp()
            logits   = scale * img_f @ clip_text_feats.T
        all_preds.append(logits.argmax(1).cpu())

    preds      = torch.cat(all_preds).numpy()
    content_np = content_labels.numpy()
    style_np   = style_labels.numpy()

    n_shape   = int((preds == content_np).sum())
    n_texture = int(((preds == style_np) & (preds != content_np)).sum())
    n_other   = len(preds) - n_shape - n_texture
    n_total   = len(preds)

    shape_bias = 100 * n_shape / (n_shape + n_texture) if (n_shape + n_texture) > 0 else 0.0
    coverage   = 100 * (n_shape + n_texture) / n_total

    return {"n_shape": n_shape, "n_texture": n_texture,
            "n_other": n_other, "n_total": n_total,
            "shape_bias": shape_bias, "coverage": coverage,
            "predictions": preds}

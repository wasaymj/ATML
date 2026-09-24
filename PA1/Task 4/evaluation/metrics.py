import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics(known_scores, unknown_scores, threshold):
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores.numpy(), unknown_scores.numpy()])
    auroc = roc_auc_score(y_true, y_scores)
    fpr95 = (unknown_scores <= threshold).float().mean().item()
    return {'auroc': auroc, 'fpr95': fpr95}

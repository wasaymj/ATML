import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
from models.resnet_cifar import SplitResNet

class PROSERMethod:
    def __init__(
        self,
        device: torch.device,
        vanilla_state_dict: dict,
        num_known_classes: int = 10,
        num_dummy_classes: int = 5,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        epochs: int = 50,
    ):
        self.device = device
        self.num_known = num_known_classes
        self.num_dummy = num_dummy_classes
        
        # Initialize PROSER model with 15 classes
        self.model = SplitResNet(num_classes=num_known_classes + num_dummy_classes).to(device)
        
        # Load Vanilla weights for everything except the extra dummy classes
        sd = self.model.state_dict()
        for k, v in vanilla_state_dict["model"].items():
            if k == "model.fc.weight":
                sd[k][:num_known_classes] = v
                # dummy weights remain randomly initialized
            elif k == "model.fc.bias":
                sd[k][:num_known_classes] = v
            else:
                sd[k] = v
        self.model.load_state_dict(sd)

        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        self.history: Dict[str, list] = {"loss": [], "loss_ce": [], "loss_cp": [], "loss_dp": []}

    def train_step(self, imgs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        
        # Split batch into two halves
        half = len(imgs) // 2
        x_cp, y_cp = imgs[:half], labels[:half]
        x_dp, y_dp = imgs[half:], labels[half:]
        
        # --- 1. CP Loss ---
        logits_cp = self.model(x_cp)
        logits_known = logits_cp[:, :self.num_known]
        logits_dummy = logits_cp[:, self.num_known:]
        
        loss_ce = F.cross_entropy(logits_known, y_cp)
        
        max_dummy, _ = logits_dummy.max(dim=1, keepdim=True)
        cp_logits_11 = torch.cat([logits_known, max_dummy], dim=1)
        
        one_hot_mask = torch.zeros(len(cp_logits_11), 11, dtype=torch.bool, device=self.device)
        one_hot_mask.scatter_(1, y_cp.unsqueeze(1), True)
        cp_logits_11 = cp_logits_11.masked_fill(one_hot_mask, -1e9)
        
        target_11 = torch.full_like(y_cp, self.num_known)
        loss_cp = F.cross_entropy(cp_logits_11, target_11)
        
        # --- 2. DP Loss ---
        if len(x_dp) > 0:
            from methods.manifold_mixup import manifold_mixup
            feats2 = self.model.forward_layer2(x_dp)
            mixed_feats2, _ = manifold_mixup(feats2, y_dp, alpha=2.0, device=self.device)
            dp_logits, _ = self.model.forward_from_layer3(mixed_feats2)
            
            dp_logits_known = dp_logits[:, :self.num_known]
            dp_max_dummy, _ = dp_logits[:, self.num_known:].max(dim=1, keepdim=True)
            dp_logits_11 = torch.cat([dp_logits_known, dp_max_dummy], dim=1)
            
            target_11_dp = torch.full_like(y_dp, self.num_known)
            loss_dp = F.cross_entropy(dp_logits_11, target_11_dp)
        else:
            loss_dp = torch.tensor(0.0).to(self.device)

        total_loss = loss_ce + 1.0 * loss_cp + 0.1 * loss_dp
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_cp": loss_cp.item(),
            "loss_dp": loss_dp.item(),
        }
        
    def step_scheduler(self):
        self.scheduler.step()

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"])

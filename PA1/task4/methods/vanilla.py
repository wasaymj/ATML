import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
from models.resnet_cifar import SplitResNet

class VanillaMethod:
    def __init__(
        self,
        device: torch.device,
        num_classes: int = 10,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        epochs: int = 100,
    ):
        self.device = device
        self.model = SplitResNet(num_classes=num_classes).to(device)
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.history: Dict[str, list] = {"loss": [], "acc": []}

    def train_step(self, imgs: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def step_scheduler(self):
        self.scheduler.step()

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"])
